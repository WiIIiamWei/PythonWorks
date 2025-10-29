# Please note that resources are .gitignored

import turtle as t

NAME = 0
POINTS = 1
POP = 2
state = ["COLORADO", [[-109, 37], [-109, 41], [-102, 41], [-102, 37]], 5187582]

cities = []
cities.append(["DENVER", [-104.98, 39.74], 634265]) 
cities.append(["BOULDER", [-105.27, 40.02], 98889]) 
cities.append(["DURANGO", [-107.88, 37.28], 17069]) 

map_width = 400
map_height = 300

minx = 180
maxx = -180
miny = 90
maxy = -90
for x,y in state[POINTS]:
    if x < minx: minx = x
    elif x > maxx: maxx = x
    if y < miny: miny = y
    elif y > maxy: maxy = y 

dist_x = maxx - minx
dist_y = maxy - miny
x_ratio = map_width / dist_x
y_ratio = map_height / dist_y

def convert(point):
    lon = point[0]
    lat = point[1]
    x = map_width - ((maxx - lon) * x_ratio)
    y = map_height - ((maxy - lat) * y_ratio)
    # Offset turtle's center origin
    x = x - (map_width/2)
    y = y - (map_height/2)
    return [x,y]

t.up()
first_pixel = None
for point in state[POINTS]:
    pixel = convert(point)
    if not first_pixel:
        first_pixel = pixel
    t.goto(pixel)
    t.down()
t.goto(first_pixel)
t.up()
t.goto([0,0])
t.write(state[NAME], align="center", font=("Arial", 16, "bold"))

for city in cities:
    pixel = convert(city[POINTS])
    t.up()
    t.goto(pixel)
    t.dot(10)
    t.write(city[NAME] + ", Pop.: " + str(city[POP]), align="left")
    t.up()

### HOMEWORK TASKS ###
# Name the map
t.goto(0, map_height/2 + 10)
t.write("B23100128魏铼", align="center", font=("Noto Sans CJK SC", 20, "bold"))

# Change a place name
for city in cities:
    if city[NAME] == "BOULDER":
        city[NAME] = "NEW CITY NAME I'M LAZY TO THINK"
        pixel = convert(city[POINTS])
        t.goto(pixel[0] + 10, pixel[1] - 10)
        t.write(city[NAME], align="left")
        break

# Change a coordinate
for city in cities:
    if city[NAME] == "DURANGO":
        city[POINTS] = [-107.88, 37.5]  # Move it slightly north
        pixel = convert(city[POINTS])
        t.goto(pixel)
        t.dot(10, "red")
        t.goto(pixel[0] + 10, pixel[1] - 10)
        t.write(city[NAME], align="left")
        break

# North-most city query
northern_city = max(cities, key=lambda city: city[POINTS][1])
t.goto(0, -200)
t.write("The northern-most city is: " + northern_city[NAME])

# Fewest population query
smallest_city = min(cities, key=lambda city: city[POP])
t.goto(0, -220)
t.write("The smallest city is: " + smallest_city[NAME])

t.pen(shown=False)
t.done()