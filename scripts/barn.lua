-- Draws the animals in each barn: one for every animal of the species its recipe uses (cows,
-- pigs, sheep, hens) that is in its input or in the craft in progress (animals waiting in the
-- output are not shown), up to the number of rendered positions. Bulls, boars and rams have
-- their own look and take the first positions. While eggs hatch, a hen broods on a nest for
-- each egg. The animals move while the barn works and hold still when it stops. The pictures
-- are rendered by blender/render_barn.py.

local sprites = require("prototypes.barn-sprites")

local lib = {}

-- Per species: the sprite kind for females (and young) and for males, and the items that count
-- for each. Bulls, boars, rams and all young animals only exist with spoiling enabled.
local SPECIES = {
    { female = "cow", male = "bull", females = { "cow", "calf-cow", "calf-bull" }, males = { "bull" } },
    { female = "pig", male = "boar", females = { "pig", "piglet-pig", "piglet-boar" }, males = { "boar" } },
    { female = "sheep", male = "ram", females = { "sheep", "lamb-sheep", "lamb-ram" }, males = { "ram" } },
    { female = "chicken", females = { "chicken", "chick" }, males = {} },
    { female = "nest", females = { "egg" }, males = {} },     -- eggs being hatched
}
local ANIMATION_PREFIX = "baketorio-retexture-barn-"   -- see prototypes/barn-animals.lua

-- Frames per tick, like vanilla's chemical plant: the 48-frame loop lasts 1.6 s.
local SPEED = 0.5

-- How often barns are checked; inserters and crafting change their contents without events.
lib.checkTicks = 30

-- Per kind: positions in drawing order (further north first, so animals further south are
-- drawn in front) and how many there are. A male kind shares its female kind's positions.
local drawOrder, positions, rendered = {}, {}, {}
for _, kind in ipairs(sprites.animals) do
    local list = require("prototypes.barn-sprites-" .. kind)
    local order = {}
    for i in ipairs(list) do order[#order + 1] = i end
    table.sort(order, function (a, b) return list[a].depth > list[b].depth end)
    drawOrder[kind], positions[kind], rendered[kind] = order, #list, true
end

-- Species items that exist in this game (get_item_count fails for unknown names).
local existing
---@type fun(names:string[]):string[]
local present = function (names)
    existing = existing or {}
    local out = {}
    for _, name in ipairs(names) do
        if existing[name] == nil then existing[name] = prototypes.item[name] ~= nil end
        if existing[name] then out[#out + 1] = name end
    end
    return out
end

-- The species a recipe works with, or nil.
---@type fun(recipe:LuaRecipe|nil):table|nil
local speciesOf = function (recipe)
    if not recipe then return nil end
    for _, sp in ipairs(SPECIES) do
        if rendered[sp.female] then
            for _, ing in pairs(recipe.ingredients) do
                for _, list in ipairs({ sp.females, sp.males }) do
                    for _, name in ipairs(list) do
                        if ing.name == name then return sp end
                    end
                end
            end
        end
    end
end

-- How many of the given items are in the barn: in its input or used by the craft in progress.
---@type fun(entity:LuaEntity, names:string[]):integer
local countItems = function (entity, names)
    local count = 0
    local input = entity.get_inventory(defines.inventory.crafter_input)
    if input then
        for _, name in ipairs(present(names)) do
            count = count + input.get_item_count(name)
        end
    end
    local recipe = entity.is_crafting() and entity.get_recipe()
    if recipe then
        for _, ing in pairs(recipe.ingredients) do
            for _, name in ipairs(names) do
                if ing.name == name then count = count + ing.amount end
            end
        end
    end
    return count
end

-- (females, males) to draw for a species, together at most its number of positions.
---@type fun(entity:LuaEntity, sp:table|nil):integer, integer
local animalCounts = function (entity, sp)
    if not sp then return 0, 0 end
    local room = positions[sp.female]
    local males = sp.male and rendered[sp.male] and countItems(entity, sp.males) or 0
    local females = countItems(entity, sp.females)
    if not (sp.male and rendered[sp.male]) then           -- no male look: draw them as females
        females = females + countItems(entity, sp.males)
    end
    males = math.min(males, room)
    return math.min(females, room - males), males
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
    local sp = speciesOf(entity.get_recipe())
    local females, males = animalCounts(entity, sp)
    local key = sp and (sp.female .. ":" .. females .. ":" .. males) or ""
    if key ~= data.key then
        for _, obj in pairs(data.renders) do
            if obj.valid then obj.destroy() end
        end
        data.renders = {}
        -- On the object layer, sorted with the barn like an entity, so characters and buildings
        -- in front of it are drawn over the animals. All shadows first, so no shadow is ever
        -- drawn over an animal. Males take the first positions, females the next ones.
        for _, suffix in ipairs({ "-shadow", "" }) do
            for _, i in ipairs(sp and drawOrder[sp.female] or {}) do
                local kind = (i <= males and sp.male) or (i <= males + females and sp.female) or nil
                if kind then
                    data.renders[#data.renders + 1] = rendering.draw_animation {
                        animation = ANIMATION_PREFIX .. kind .. "-" .. i .. suffix,
                        target = entity,
                        surface = entity.surface,
                        render_layer = "object",
                        animation_speed = data.speed,
                        animation_offset = data.offset,
                    }
                end
            end
        end
        data.key = key
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
        data = { entity = entity, renders = {}, key = "", speed = 0,
                 offset = (entity.unit_number * 7) % sprites.frames, moving = false }
        storage.barns[entity.unit_number] = data
    end
    updateBarn(data)
end

-- Finds every barn again, e.g. after the mod is added to an existing save.
-- storage.barns: {LuaEntity.unit_number: {entity, renders: LuaRenderObject[], key, speed, offset, moving}}
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
