local greenhouse = require("scripts.greenhouse")

script.on_init(function ()
    greenhouse.rebuildGreenhouses()
end
)

script.on_configuration_changed(function (confData)
    greenhouse.rebuildGreenhouses()
end
)

-- Track greenhouses so the crop matching their recipe and harvest can be drawn inside
local greenhouseFilter = { { filter = "name", name = "greenhouse" } }

script.on_event(defines.events.on_built_entity, function (event)
    greenhouse.trackGreenhouse(event.entity)
end, greenhouseFilter)

script.on_event(defines.events.on_robot_built_entity, function (event)
    greenhouse.trackGreenhouse(event.entity)
end, greenhouseFilter)

script.on_event(defines.events.script_raised_built, function (event)
    greenhouse.trackGreenhouse(event.entity)
end, greenhouseFilter)

script.on_event(defines.events.script_raised_revive, function (event)
    greenhouse.trackGreenhouse(event.entity)
end, greenhouseFilter)

script.on_event(defines.events.on_entity_cloned, function (event)
    greenhouse.trackGreenhouse(event.destination)
end, greenhouseFilter)

script.on_event(defines.events.on_player_rotated_entity, function (event)
    greenhouse.trackGreenhouse(event.entity)
end)

script.on_nth_tick(greenhouse.checkTicks, function ()
    greenhouse.updateAllGreenhouses()
end)
