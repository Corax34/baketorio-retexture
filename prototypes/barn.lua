-- Baketorio+'s barn as a cow pasture (rendered by blender/render_barn.py): double its 3x3
-- size. The barn itself shows the empty paddock; scripts/barn.lua draws the cows in it,
-- as many as the barn holds.

local util = require("util")
local png = require("prototypes.paths").png
local sprites = require("prototypes.barn-sprites")
local fluidBoxes = require("prototypes.fluid-boxes")

---@type fun(s:table, file:string, extra:table|nil):table
local layer = function (s, file, extra)
    local l = {
        filename = png(file),
        width = s.width,
        height = s.height,
        shift = s.shift,
        scale = 0.5,
        frame_count = 1,
    }
    for k, v in pairs(extra or {}) do l[k] = v end
    return l
end

local barn = data.raw["assembling-machine"]["barn"]
if barn then
    local size = sprites.size
    barn.graphics_set = barn.graphics_set or {}
    barn.graphics_set.animation = {
        layers = {
            layer(sprites.base, "barn-base"),
            layer(sprites.shadow, "barn-shadow", { draw_as_shadow = true }),
        }
    }
    barn.collision_box = { { -size / 2 + 0.2, -size / 2 + 0.2 }, { size / 2 - 0.2, size / 2 - 0.2 } }
    barn.selection_box = { { -size / 2, -size / 2 }, { size / 2, size / 2 } }
    barn.drawing_box = nil
    barn.drawing_box_vertical_extension = 1.5
    barn.icon_draw_specification = { shift = { 0, -0.3 }, scale = 1.5 }
    barn.alert_icon_shift = util.by_pixel(0, -20)
    fluidBoxes.fitToEdges(barn, size, size)
end
