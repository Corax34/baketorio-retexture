-- Points every prototype that uses one of Baketorio's replaced icon files at the HD
-- version, wherever it is used (items, recipes, technologies, icon layers of other mods).
-- Layer sizes keep looking the same: Factorio scales an icon layer by its icon_size.

local png = require("prototypes.paths").png

-- Baketorio file -> { replacement, its icon_size }
local replaced = {
    ["__baketorio__/graphics/salt.png"]            = { png("salt"), 64 },
    ["__baketorio__/graphics/trona.png"]           = { png("trona"), 64 },
    ["__baketorio__/graphics/salt-mod.png"]        = { png("salt-mod"), 64 },
    ["__baketorio__/graphics/trona_tech.png"]      = { png("trona_tech"), 256 },
    ["__baketorio__/graphics/greenhouse-icon.png"] = { png("greenhouse-icon"), 64 },
}

for _, prototypes in pairs(data.raw) do
    for _, proto in pairs(prototypes) do
        if type(proto) == "table" then
            local new = proto.icon and replaced[proto.icon]
            if new then
                proto.icon, proto.icon_size = new[1], new[2]
            end
            if type(proto.icons) == "table" then
                for _, layer in pairs(proto.icons) do
                    local new_layer = type(layer) == "table" and layer.icon and replaced[layer.icon]
                    if new_layer then
                        layer.icon, layer.icon_size = new_layer[1], new_layer[2]
                    end
                end
            end
        end
    end
end
