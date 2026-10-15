-- TEMPORARY: remove once Baketorio+ is updated for Factorio 2.1.21.
-- Factorio 2.1 removed ItemPrototype::fuel_category in favour of fuel_categories, but
-- Baketorio+ 0.6.1 still sets the old field (on butter), so the game refuses to load.
-- Move the old field into the new list.

for _, prototypes in pairs(data.raw) do
    for _, proto in pairs(prototypes) do
        if type(proto) == "table" and proto.fuel_category then
            local categories = proto.fuel_categories or {}
            local present = false
            for _, c in pairs(categories) do
                if c == proto.fuel_category then present = true end
            end
            if not present then table.insert(categories, proto.fuel_category) end
            proto.fuel_categories = categories
            proto.fuel_category = nil
        end
    end
end
