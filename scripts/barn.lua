-- Draws the animals in each barn: one for every animal of the kind its recipe uses (cows,
-- pigs) that is in its input or in the craft in progress (animals waiting in the output are
-- not shown), up to the number of rendered positions. They move while the barn works and
-- hold still when it stops. The pictures are rendered by blender/render_barn.py.

local sprites = require("prototypes.barn-sprites")

local lib = {}

-- Items that count as each animal kind (calves and piglets only exist with spoiling enabled).
local ANIMAL_ITEMS = {
    cow = { "cow", "bull", "calf-cow", "calf-bull" },
    pig = { "pig", "boar", "piglet-pig", "piglet-boar" },
}
local ANIMATION_PREFIX = "baketorio-retexture-barn-"   -- see prototypes/barn-animals.lua

-- Frames per tick, like vanilla's chemical plant: the 48-frame loop lasts 1.6 s.
local SPEED = 0.5

-- How often barns are checked; inserters and crafting change their contents without events.
lib.checkTicks = 30

-- Per kind: positions in drawing order (further north first, so animals further south are
-- drawn in front) and how many there are.
local drawOrder, positions = {}, {}
for _, kind in ipairs(sprites.animals) do
    local list = require("prototypes.barn-sprites-" .. kind)
    local order = {}
    for i in ipairs(list) do order[#order + 1] = i end
    table.sort(order, function (a, b) return list[a].depth > list[b].depth end)
    drawOrder[kind], positions[kind] = order, #list
end

-- Animal items of each kind that exist in this game.
local items
---@type fun(kind:string):string[]
local itemsOf = function (kind)
    if not items then
        items = {}
        for k, names in pairs(ANIMAL_ITEMS) do
            items[k] = {}
            for _, name in ipairs(names) do
                if prototypes.item[name] then items[k][#items[k] + 1] = name end
            end
        end
    end
    return items[kind] or {}
end

-- The animal kind a recipe works with, or nil.
---@type fun(recipe:LuaRecipe|nil):string|nil
local kindOf = function (recipe)
    if not recipe then return nil end
    for _, kind in ipairs(sprites.animals) do
        for _, ing in pairs(recipe.ingredients) do
            for _, name in ipairs(itemsOf(kind)) do
                if ing.name == name then return kind end
            end
        end
    end
end

---@type fun(entity:LuaEntity, kind:string|nil):integer
local animalCount = function (entity, kind)
    if not kind then return 0 end
    local count = 0
    local input = entity.get_inventory(defines.inventory.crafter_input)
    if input then
        for _, name in ipairs(itemsOf(kind)) do
            count = count + input.get_item_count(name)
        end
    end
    -- Animals used by the craft in progress have left the input but are still in the barn.
    local recipe = entity.is_crafting() and entity.get_recipe()
    if recipe then
        for _, ing in pairs(recipe.ingredients) do
            for _, name in ipairs(itemsOf(kind)) do
                if ing.name == name then count = count + ing.amount end
            end
        end
    end
    return math.min(positions[kind], count)
end

-- Each barn keeps its own animation clock (frame = tick * speed + offset), so animals can
-- be added or removed, and stopped or started, without jumping to another frame.
---@type fun(data:table, moving:boolean)
local setMoving = function (data, moving)
    local frame = (game.tick * data.speed + data.offset) % sprites.frames
    data.speed = moving and SPEED or 0
    data.offset = frame - game.tick * data.speed
    data.moving = moving
    for _, obj in pairs(data.renders) do
        if obj.valid then
            obj.animation_speed = data.speed
            obj.animation_offset = data.offset
        end
    end
end

---@type fun(data:table)
local updateBarn = function (data)
    local entity = data.entity
    local kind = kindOf(entity.get_recipe())
    local count = animalCount(entity, kind)
    if count ~= data.count or kind ~= data.kind then
        for _, obj in pairs(data.renders) do
            if obj.valid then obj.destroy() end
        end
        data.renders = {}
        -- Shadows on a lower layer than the animals, so no shadow is ever drawn over an animal.
        for _, part in ipairs({ { "-shadow", "higher-object-under" }, { "", "higher-object-above" } }) do
            for _, i in ipairs(kind and drawOrder[kind] or {}) do
                if i <= count then
                    data.renders[#data.renders + 1] = rendering.draw_animation {
                        animation = ANIMATION_PREFIX .. kind .. "-" .. i .. part[1],
                        target = entity,
                        surface = entity.surface,
                        render_layer = part[2],
                        animation_speed = data.speed,
                        animation_offset = data.offset,
                    }
                end
            end
        end
        data.count, data.kind = count, kind
    end
    local moving = entity.status == defines.entity_status.working
    if moving ~= data.moving then setMoving(data, moving) end
end

-- Starts tracking a barn (built, cloned); ignores other entities.
---@type fun(entity:LuaEntity|nil)
lib.trackBarn = function (entity)
    if not (entity and entity.valid and entity.name == "barn") then return end
    storage.barns = storage.barns or {}
    local data = storage.barns[entity.unit_number]
    if not data then
        -- Start barns at different points in the loop so neighbours don't move in step.
        data = { entity = entity, renders = {}, count = 0, speed = 0,
                 offset = (entity.unit_number * 7) % sprites.frames, moving = false }
        storage.barns[entity.unit_number] = data
    end
    updateBarn(data)
end

-- Finds every barn again, e.g. after the mod is added to an existing save.
-- storage.barns: {LuaEntity.unit_number: {entity, renders: LuaRenderObject[], kind, count, speed, offset, moving}}
---@type fun()
lib.rebuildBarns = function ()
    for _, obj in pairs(rendering.get_all_objects("baketorio-retexture")) do
        if obj.type == "animation" and obj.animation:find(ANIMATION_PREFIX, 1, true) == 1 then obj.destroy() end
    end
    storage.barns = {}
    if not prototypes.entity["barn"] then return end
    for _, surface in pairs(game.surfaces) do
        for _, entity in pairs(surface.find_entities_filtered { name = "barn" }) do
            lib.trackBarn(entity)
        end
    end
end

---@type fun()
lib.updateAllBarns = function ()
    for unit, data in pairs(storage.barns or {}) do
        if data.entity.valid then
            updateBarn(data)
        else
            storage.barns[unit] = nil
        end
    end
end

return lib
