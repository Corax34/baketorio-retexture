-- HD salt and trona: 128 px ore sheets with 8 stages x 8 variations, 64 px icons,
-- and 4 picture variants for the items on belts and the ground.

local png = require("prototypes.paths").png

for _, name in ipairs({ "salt", "trona" }) do
    local resource = data.raw["resource"][name]
    if resource then
        resource.icon = png(name)
        resource.icon_size = 64
        resource.icons = nil
        resource.stage_counts = { 1000, 750, 550, 400, 280, 180, 100, 50 }
        resource.stages = {
            sheet = {
                filename = png(name .. "-ore"),
                priority = "extra-high",
                size = 128,
                frame_count = 8,
                variation_count = 8,
                scale = 0.5
            }
        }
    end

    local item = data.raw["item"][name]
    if item then
        item.icon = png(name)
        item.icon_size = 64
        item.icons = nil
        item.pictures = {
            { size = 64, filename = png(name),        scale = 0.5, mipmap_count = 4 },
            { size = 64, filename = png(name .. "-1"), scale = 0.5, mipmap_count = 4 },
            { size = 64, filename = png(name .. "-2"), scale = 0.5, mipmap_count = 4 },
            { size = 64, filename = png(name .. "-3"), scale = 0.5, mipmap_count = 4 },
        }
    end
end
