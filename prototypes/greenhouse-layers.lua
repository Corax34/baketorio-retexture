-- Sprite layers for the greenhouse rendered by blender/render_greenhouse.py: one picture
-- for north/south and one for east/west, plus per-crop pictures that control.lua draws inside.

local png = require("prototypes.paths").png
local sprites = require("prototypes.greenhouse-sprites")

local lib = {}

lib.crops = { "wheat", "sugarcane", "cocoa", "cinnamon", "blueberries", "strawberries",
              "apples", "peaches", "lemons", "nuts", "pumpkin", "wood" }
lib.stages = 8
lib.spritePrefix = "baketorio-retexture-greenhouse-"

---@type fun(name:string, file:string, extra:table|nil):table
lib.layer = function (name, file, extra)
    local s = sprites[name]
    local layer = {
        filename = png(file),
        width = s.width,
        height = s.height,
        frame_count = 1,
        scale = 0.5,
        shift = s.shift,
    }
    for k, v in pairs(extra or {}) do layer[k] = v end
    return layer
end

---@type fun(orient:string):table
lib.direction = function (orient)
    return {
        layers = {
            lib.layer("main-" .. orient, "greenhouse-" .. orient),
            lib.layer("shadow-" .. orient, "greenhouse-" .. orient .. "-shadow", { draw_as_shadow = true }),
        }
    }
end

return lib
