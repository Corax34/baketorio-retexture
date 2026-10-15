-- Fixes for other mods, after every mod has run its data-updates.lua.

if mods["baketorio_plus"] then
    require("prototypes.compat-baketorio-plus")   -- temporary, see the file
end
