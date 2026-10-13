-- Crop pictures drawn inside greenhouses by scripts/greenhouse.lua: per crop and
-- orientation, harvest stages 0 (empty beds) to 8 (full).

local layers = require("prototypes.greenhouse-layers")

for _, crop in ipairs(layers.crops) do
    for _, orient in ipairs({ "ns", "ew" }) do
        for stage = 0, layers.stages do
            local name = crop .. "-" .. orient .. "-" .. stage
            local sprite = layers.layer(name, "greenhouse-" .. name)
            sprite.type = "sprite"
            sprite.name = layers.spritePrefix .. name
            sprite.frame_count = nil
            data:extend({ sprite })
        end
    end
end
