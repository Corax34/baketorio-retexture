local greenhouse = require("scripts.greenhouse")
local barn = require("scripts.barn")

script.on_init(function ()
    greenhouse.rebuildGreenhouses()
    barn.rebuildBarns()
end
)

script.on_configuration_changed(function (confData)
    greenhouse.rebuildGreenhouses()
    barn.rebuildBarns()
end
)

-- Track greenhouses and barns so their crops and cows can be drawn inside
local buildingFilter = { { filter = "name", name = "greenhouse" }, { filter = "name", name = "barn" } }

---@type fun(entity:LuaEntity|nil)
local track = function (entity)
    greenhouse.trackGreenhouse(entity)
    barn.trackBarn(entity)
end

script.on_event(defines.events.on_built_entity, function (event)
    track(event.entity)
end, buildingFilter)

script.on_event(defines.events.on_robot_built_entity, function (event)
    track(event.entity)
end, buildingFilter)

script.on_event(defines.events.script_raised_built, function (event)
    track(event.entity)
end, buildingFilter)

script.on_event(defines.events.script_raised_revive, function (event)
    track(event.entity)
end, buildingFilter)

script.on_event(defines.events.on_entity_cloned, function (event)
    track(event.destination)
end, buildingFilter)

script.on_event(defines.events.on_player_rotated_entity, function (event)
    greenhouse.trackGreenhouse(event.entity)
end)

script.on_nth_tick(greenhouse.checkTicks, function ()
    greenhouse.updateAllGreenhouses()
end)

script.on_nth_tick(barn.checkTicks, function ()
    barn.updateAllBarns()
end)
