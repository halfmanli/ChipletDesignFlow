from .. import dataset

bdg_all = dataset.AMD()
area_all = []
for bdg in bdg_all:
    area = 0
    for _, v_attr in bdg.nodes(data=True):
        area += v_attr["block"].area
    area_all.append(area)

print(sum(area_all) / len(area_all))
