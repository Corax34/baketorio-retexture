-- Two animations per animal position in the barn, drawn by scripts/barn.lua as animals
-- arrive: the animal and its shadow, each a still image of everything that never moves
-- plus small animated patches for the head, ears and tail (rendered by blender/render_barn.py).

local png = require("prototypes.paths").png
local sprites = require("prototypes.barn-sprites")

local ANIMATION_PREFIX = "baketorio-retexture-barn-"   -- see scripts/barn.lua

---@type fun(s:table, frameCount:integer, extra:table):table
local layer = function (s, frameCount, extra)
    local l = {
        filename = png(s.file),
        width = s.width,
        height = s.height,
        shift = s.shift,
        scale = 0.5,
        frame_count = frameCount,
    }
    for k, v in pairs(extra) do l[k] = v end
    return l
end

---@type fun(part:table, name:string)
local addAnimation = function (part, name)
    local layers = {}
    if part.still then
        -- The still image repeats its one frame so every layer loops over the same length.
        layers[#layers + 1] = layer(part.still, 1, { repeat_count = sprites.frames })
    end
    for _, patch in ipairs(part.patches) do
        layers[#layers + 1] = layer(patch, sprites.frames, { line_length = patch.line_length })
    end
    data:extend({ { type = "animation", name = name, layers = layers } })
end

-- Only with Baketorio+, which adds the barn (its data.lua runs before this one).
if data.raw["assembling-machine"]["barn"] then
    for _, kind in ipairs(sprites.animals) do
        for i, animal in ipairs(require("prototypes.barn-sprites-" .. kind)) do
            addAnimation(animal.body, ANIMATION_PREFIX .. kind .. "-" .. i)
            addAnimation(animal.shadow, ANIMATION_PREFIX .. kind .. "-" .. i .. "-shadow")
        end
    end
end
