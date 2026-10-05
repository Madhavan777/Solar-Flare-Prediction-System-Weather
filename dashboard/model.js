/* ------------------------------------------------------------------------- *
 * The selected model, reimplemented for the browser.
 *
 * This file is the single source of truth for the client-side pipeline. The
 * dashboard loads it with a <script> tag; tests/test_js_parity.py loads the
 * same file in Node and checks it against scikit-learn on real data:
 *
 *   - probabilities on >= 1,000 precomputed P5 feature rows, max abs diff < 1e-6
 *   - feature extraction on >= 200 real SWAN-SF windows, max abs diff < 1e-6
 *
 * Every fitted parameter is read from model_lr.json, which is exported
 * verbatim from models/I2_LR_temporal_C0.01.joblib. Nothing here is fitted,
 * rounded or hard-coded.
 * ------------------------------------------------------------------------- */
(function (root) {
  'use strict';

  /* The fixed signed-log transform. Not learned, so it cannot leak. */
  function slog(x) {
    return Math.sign(x) * Math.log10(1 + Math.abs(x));
  }

  /**
   * Summarise one observation window as the model's feature vector.
   *
   * Mirrors solarflare.features.window_features exactly, including the edge
   * cases that matter on real data:
   *   - statistics use only the finite values of each parameter;
   *   - "last" is the last FINITE value, not the final row;
   *   - "std" is the population standard deviation (ddof = 0);
   *   - "slope" is a least-squares trend per hour over the finite points, on a
   *     time axis of index x cadence_hours, and is NaN unless at least two
   *     finite points exist that span more than an instant;
   *   - a parameter that is entirely missing yields NaN for all six of its
   *     statistics, left for the imputer.
   *
   * @param {Object} columns  {PARAM: [v, ...]} with null for missing.
   * @param {Object} model    the parsed model_lr.json.
   * @returns {number[]} the feature vector, ordered as model.cols.
   */
  function windowFeatures(columns, model) {
    var params = model.sharp_parameters;
    var stats = model.statistics;
    var out = new Array(params.length * stats.length).fill(NaN);

    for (var j = 0; j < params.length; j++) {
      var col = columns[params[j]] || [];
      var vals = [];
      var times = [];
      for (var i = 0; i < col.length; i++) {
        var v = col[i];
        if (v !== null && v !== undefined && Number.isFinite(v)) {
          vals.push(v);
          times.push(i * model.cadence_hours);
        }
      }
      var base = j * stats.length;
      if (vals.length === 0) continue;

      var sum = 0, min = Infinity, max = -Infinity;
      for (var k = 0; k < vals.length; k++) {
        sum += vals[k];
        if (vals[k] < min) min = vals[k];
        if (vals[k] > max) max = vals[k];
      }
      var mean = sum / vals.length;
      var varsum = 0;
      for (var m = 0; m < vals.length; m++) varsum += (vals[m] - mean) * (vals[m] - mean);

      out[base + 0] = vals[vals.length - 1];
      out[base + 1] = mean;
      out[base + 2] = Math.sqrt(varsum / vals.length);
      out[base + 3] = min;
      out[base + 4] = max;

      if (vals.length >= 2) {
        var tmin = times[0], tmax = times[times.length - 1];
        if (tmax > tmin) {
          var st = 0, sy = 0, stt = 0, sty = 0, n = vals.length;
          for (var q = 0; q < n; q++) {
            st += times[q];
            sy += vals[q];
            stt += times[q] * times[q];
            sty += times[q] * vals[q];
          }
          var denom = n * stt - st * st;
          if (denom !== 0) out[base + 5] = (n * sty - st * sy) / denom;
        }
      }
    }
    return out;
  }

  /**
   * Run the fitted pipeline: signed-log, median impute, standardise, logistic.
   *
   * @param {number[]} features  a feature vector ordered as model.cols.
   * @param {Object} model       the parsed model_lr.json.
   * @returns {{probability:number, logit:number, z:number[], contrib:number[]}}
   *   The contributions plus the intercept sum to the logit exactly.
   */
  function predict(features, model) {
    var n = model.cols.length;
    var z = new Array(n);
    var contrib = new Array(n);
    var logit = model.intercept;

    for (var j = 0; j < n; j++) {
      var v = features[j];
      if (model.apply_slog && Number.isFinite(v)) v = slog(v);
      if (!Number.isFinite(v)) v = model.impute_median[j];
      var s = (v - model.scale_mean[j]) / model.scale_std[j];
      z[j] = s;
      contrib[j] = s * model.coef[j];
      logit += contrib[j];
    }
    return { probability: 1 / (1 + Math.exp(-logit)), logit: logit, z: z, contrib: contrib };
  }

  /**
   * Parse a SWAN-SF window file (tab-separated, header row, 60 data rows).
   *
   * @throws {Error} if a required SHARP column is absent.
   */
  function parseWindowCSV(text, model) {
    var lines = text.split(/\r?\n/).filter(function (l) { return l.trim().length; });
    if (lines.length < 2) throw new Error('the file has no data rows');

    var sep = lines[0].indexOf('\t') >= 0 ? '\t' : (lines[0].indexOf(',') >= 0 ? ',' : /\s+/);
    var header = lines[0].split(sep).map(function (s) {
      return s.trim().replace(/^"|"$/g, '');
    });

    var index = {};
    for (var a = 0; a < model.sharp_parameters.length; a++) {
      var p = model.sharp_parameters[a];
      var at = header.indexOf(p);
      if (at < 0) throw new Error('column ' + p + ' is missing from the header');
      index[p] = at;
    }

    var columns = {};
    for (var b = 0; b < model.sharp_parameters.length; b++) columns[model.sharp_parameters[b]] = [];

    var rows = 0;
    for (var i = 1; i < lines.length; i++) {
      var cells = lines[i].split(sep);
      if (cells.length < header.length) continue;
      rows++;
      for (var c = 0; c < model.sharp_parameters.length; c++) {
        var name = model.sharp_parameters[c];
        var raw = (cells[index[name]] || '').trim();
        var num = raw === '' ? null : Number(raw);
        columns[name].push(raw === '' || !Number.isFinite(num) ? null : num);
      }
    }
    return { columns: columns, rows: rows };
  }

  var api = {
    slog: slog,
    windowFeatures: windowFeatures,
    predict: predict,
    parseWindowCSV: parseWindowCSV
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;          // Node, for the parity tests
  } else {
    root.SolarFlareModel = api;    // browser
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
