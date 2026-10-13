-- The 3D glass greenhouse: 5x7 tiles, one picture per orientation pair.
-- Baketorio+ builds its barn as a copy of the greenhouse in its data.lua; this runs later
-- (data-updates), so the barn keeps the original 3x3 size and picture.

local util = require("util")
local layers = require("prototypes.greenhouse-layers")

local WIDTH, LENGTH = 5, 7

-- Moves every pipe connection to the edge of the resized building, keeping its side,
-- and draws north-pointing pipes behind the building instead of in front of its wall.
local function fitFluidBoxes(entity)
    local hw, hl = (WIDTH - 1) / 2, (LENGTH - 1) / 2
    for _, fb in pairs(entity.fluid_boxes or {}) do
        for _, pc in pairs(fb.pipe_connections or {}) do
            local p = pc.position
            if p then
                local x = math.max(-hw, math.min(hw, p[1] or p.x))
                local y = math.max(-hl, math.min(hl, p[2] or p.y))
                if pc.direction == defines.direction.north then y = -hl
                elseif pc.direction == defines.direction.south then y = hl
                elseif pc.direction == defines.direction.west then x = -hw
                elseif pc.direction == defines.direction.east then x = hw
                end
                pc.position = { x, y }
            end
        end
        fb.secondary_draw_orders = fb.secondary_draw_orders or {}
        fb.secondary_draw_orders.north = -1
    end
end

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
    fitFluidBoxes(greenhouse)
end
