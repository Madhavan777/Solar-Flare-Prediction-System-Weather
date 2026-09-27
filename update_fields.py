"""Open the built docx in a running headless LibreOffice, update all fields
(TOC, PAGEREF, PAGE) and the TOC index, then save back to docx."""
import sys, uno
from com.sun.star.beans import PropertyValue


def mkprop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def main(path):
    localContext = uno.getComponentContext()
    resolver = localContext.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", localContext)
    ctx = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

    url = "file://" + path
    doc = desktop.loadComponentFromURL(url, "_blank", 0, (mkprop("Hidden", True),))

    # Update every TOC-like index first
    if doc.supportsService("com.sun.star.text.GenericTextDocument"):
        indexes = doc.getDocumentIndexes()
        for i in range(indexes.getCount()):
            idx = indexes.getByIndex(i)
            idx.update()

    # Update all text fields (PAGEREF, PAGE, etc.) - may need a couple passes
    # since PAGEREF depends on pagination settling after the TOC is inserted.
    for _ in range(3):
        doc.getTextFields().refresh()
        indexes = doc.getDocumentIndexes()
        for i in range(indexes.getCount()):
            indexes.getByIndex(i).update()

    doc.store()
    doc.close(False)
    print("fields updated and saved:", path)


if __name__ == "__main__":
    main(sys.argv[1])
