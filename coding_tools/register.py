TOOL_GROUPS = {}


def register_tool(name):
    def _func(cls):
        TOOL_GROUPS[name] = cls
        return cls

    return _func
