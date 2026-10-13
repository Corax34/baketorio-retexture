-- Draws the crop that matches each greenhouse's active recipe inside it, showing how
-- much harvest is waiting in the output: empty beds while growing, then more and more
-- plants, one stage per harvested item, full from STAGES items. The pictures are
-- rendered by blender/render_greenhouse.py.

local lib = {}

local CROPS = {
    wheat = "wheat",
    sugarcane = "sugarcane",
    sugar = "sugarcane",
    cocoa = "cocoa",
    cinnamon = "cinnamon",
    blueberry = "blueberries",
    blueberries = "blueberries",
    strawberry = "strawberries",
    strawberries = "strawberries",
    -- Baketorio+
    apple = "apples",
    apples = "apples",
    peach = "peaches",
    peaches = "peaches",
    lemon = "lemons",
    lemons = "lemons",
    nut = "nuts",
    nuts = "nuts",
    pumpkin = "pumpkin",
    wood = "wood",
}

local STAGES = 8
local SPRITE_PREFIX = "baketorio-retexture-greenhouse-"   -- see prototypes/greenhouse-crops.lua

-- How often greenhouses are checked; recipes and outputs change without events.
lib.checkTicks = 60

-- recipe name -> crop name, or false if the recipe has no crop picture
local cropCache = {}

-- recipe name -> list of the item names it harvests
local harvestCache = {}

---@type fun(name:string):string|nil
local cropInName = function (name)
    for word in name:gmatch("[^%-]+") do
        if CROPS[word] then return CROPS[word] end
    end
end

---@type fun(recipe:LuaRecipe):string|nil
local cropForRecipe = function (recipe)
    local cached = cropCache[recipe.name]
    if cached ~= nil then return cached or nil end
    local crop = cropInName(recipe.name)
    if not crop then
        -- Products only: animal recipes may use crops as feed.
        for _, item in pairs(recipe.products) do
            crop = cropInName(item.name)
            if crop then break end
        end
    end
    cropCache[recipe.name] = crop or false
    return crop
end

-- The harvest is what the recipe makes that it doesn't also use, so seeds handed
-- back by a crop recipe don't count, but the seeds made by a seed recipe do. Crops
-- that are their own seed (nuts: 2 nuts in, 20 out) have nothing else, so then
-- every item product counts.
---@type fun(recipe:LuaRecipe):string[]
local harvestItems = function (recipe)
    local cached = harvestCache[recipe.name]
    if cached then return cached end
    local used = {}
    for _, ing in pairs(recipe.ingredients) do used[ing.name] = true end
    local items, all = {}, {}
    for _, product in pairs(recipe.products) do
        if product.type == "item" then
            all[#all + 1] = product.name
            if not used[product.name] then items[#items + 1] = product.name end
        end
    end
    if #items == 0 then items = all end
    harvestCache[recipe.name] = items
    return items
end

-- One stage per harvested item waiting in the output: 0 = none, STAGES or more = full.
-- (In normal play the output is emptied by inserters and rarely holds more than a few.)
---@type fun(entity:LuaEntity, recipe:LuaRecipe):integer
local harvestStage = function (entity, recipe)
    local output = entity.get_output_inventory()
    if not output then return 0 end
    local count = 0
    for _, name in ipairs(harvestItems(recipe)) do
        count = math.max(count, output.get_item_count(name))
    end
    return math.min(STAGES, count)
end

---@type fun(entity:LuaEntity):string|nil
local spriteFor = function (entity)
    local recipe = entity.get_recipe()
    local crop = recipe and cropForRecipe(recipe)
    if not crop then return nil end
    local d = entity.direction
    local orient = (d == defines.direction.east or d == defines.direction.west) and "ew" or "ns"
    return SPRITE_PREFIX .. crop .. "-" .. orient .. "-" .. harvestStage(entity, recipe)
end

---@type fun(data:table)
local updateGreenhouse = function (data)
    local sprite = spriteFor(data.entity)
    if sprite == data.sprite then return end
    if data.render and data.render.valid then data.render.destroy() end
    data.render = nil
    data.sprite = sprite
    if sprite then
        data.render = rendering.draw_sprite {
            sprite = sprite,
            target = data.entity,
            surface = data.entity.surface,
            render_layer = "higher-object-under",
        }
    end
end

-- Starts tracking a greenhouse (built, cloned, rotated); ignores other entities.
---@type fun(entity:LuaEntity|nil)
lib.trackGreenhouse = function (entity)
    if not (entity and entity.valid and entity.name == "greenhouse") then return end
    -- A save from before this script existed has no table until the next rebuild.
    storage.greenhouses = storage.greenhouses or {}
    local data = storage.greenhouses[entity.unit_number] or { entity = entity }
    storage.greenhouses[entity.unit_number] = data
    updateGreenhouse(data)
end

-- Finds every greenhouse again, e.g. after the mod is added to an existing save.
-- storage.greenhouses: {LuaEntity.unit_number: {entity: LuaEntity, sprite: string|nil, render: LuaRenderObject|nil}}
---@type fun()
lib.rebuildGreenhouses = function ()
    for _, obj in pairs(rendering.get_all_objects("baketorio-retexture")) do
        if obj.type == "sprite" and obj.sprite:find(SPRITE_PREFIX, 1, true) == 1 then obj.destroy() end
    end
    storage.greenhouses = {}
    for _, surface in pairs(game.surfaces) do
        for _, entity in pairs(surface.find_entities_filtered { name = "greenhouse" }) do
            lib.trackGreenhouse(entity)
        end
    end
end

-- Recipes and outputs change without an event (circuits, blueprints, inserters), so poll.
---@type fun()
lib.updateAllGreenhouses = function ()
    for unit, data in pairs(storage.greenhouses or {}) do
        if data.entity.valid then
            updateGreenhouse(data)
        else
            storage.greenhouses[unit] = nil
        end
    end
end

return lib
