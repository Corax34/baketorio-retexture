local lib = {}

-- Nearest tile centre to v for a side of `size` tiles: whole numbers for odd sizes,
-- halves for even sizes (where the middle falls on a tile edge, +0.5 is used).
---@type fun(v:number, size:integer):number
local tileCentre = function (v, size)
    local half = (size - 1) / 2
    local snapped = math.floor(v + half + 0.5) - half
    return math.max(-half, math.min(half, snapped))
end

-- Moves every pipe connection to the edge of a resized building (north orientation,
-- width x length tiles), keeping its side, and draws north-pointing pipes behind the
-- building instead of in front of its back wall.
---@type fun(entity:table, width:integer, length:integer)
lib.fitToEdges = function (entity, width, length)
    local hw, hl = (width - 1) / 2, (length - 1) / 2
    for _, fb in pairs(entity.fluid_boxes or {}) do
        for _, pc in pairs(fb.pipe_connections or {}) do
            local p = pc.position
            if p then
                local x = tileCentre(p[1] or p.x, width)
                local y = tileCentre(p[2] or p.y, length)
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

return lib
