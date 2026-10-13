local lib = {}

---@type fun(name:string):string
lib.png = function (name)
    return "__baketorio-retexture__/graphics/" .. name .. ".png"
end

return lib
