def prettyPrintTree(root, ident):
    print(ident + str(root) + "  {" + str(id(root)) + "}")
    if ident == "":
        ident = "|_"
    for child in root.children:
        prettyPrintTree(child, " " + ident)
