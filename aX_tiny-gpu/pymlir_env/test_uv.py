import mlir

m = mlir.parse_path("../main_generic.mlir")

print(m.dump())

print("---")

# Dump the AST directly
print(m.dump_ast())

print("---")


# Or visit each node type by implementing visitor functions
class MyVisitor(mlir.NodeVisitor):
    def visit_Function(self, node: mlir.astnodes.Function):
        print("Function detected:", node.name.value)


MyVisitor().visit(m)
