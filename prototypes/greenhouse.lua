-- The 3D glass greenhouse: 5x7 tiles, one picture per orientation pair.
-- Baketorio+ builds its barn as a copy of the greenhouse in its data.lua; this runs later
-- (data-updates), so the barn keeps the original 3x3 size and picture.

local util = require("util")
local layers = require("prototypes.greenhouse-layers")
local fluidBoxes = require("prototypes.fluid-boxes")

local WIDTH, LENGTH = 5, 7

local greenhouse = data.raw["assembling-machine"]["greenhouse"]
if greenhouse then
    greenhouse.graphics_set = greenhouse.graphics_set or {}
    greenhouse.graphics_set.animation = {
        north = layers.direction("ns"),
        south = layers.direction("ns"),
        east = layers.direction("ew"),
        west = layers.direction("ew"),
    }
    greenhouse.collision_box = { { -WIDTH / 2 + 0.2, -LENGTH / 2 + 0.2 }, { WIDTH / 2 - 0.2, LENGTH / 2 - 0.2 } }
    greenhouse.selection_box = { { -WIDTH / 2, -LENGTH / 2 }, { WIDTH / 2, LENGTH / 2 } }
    greenhouse.drawing_box = nil
    greenhouse.drawing_box_vertical_extension = 2
    greenhouse.icon_draw_specification = { shift = { 0, -0.5 }, scale = 1.5 }
    greenhouse.alert_icon_shift = util.by_pixel(0, -24)
    greenhouse.impact_category = "glass"
    fluidBoxes.fitToEdges(greenhouse, WIDTH, LENGTH)
end
