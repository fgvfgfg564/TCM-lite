TOOL_GROUPS = {}
def register_tool(name, cls):
    TOOL_GROUPS[name] = cls