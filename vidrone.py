import math, time, heapq, random, requests, pygame, itertools, threading, queue
import numpy as np
import networkx as nx
import os
from shapely.geometry import Polygon, Point

# ======== CONFIG ========
CENTER_LAT = 12.9716
CENTER_LON = 77.5946
DRONE_SPEED_MPS = 30.0
LAYER_HEIGHT_M = 30.0
MAX_DRONE_ALT_M = 150.0
LAYERS = int(math.ceil(MAX_DRONE_ALT_M / LAYER_HEIGHT_M))
INIT_WIN_W, INIT_WIN_H = 1200, 800
BUILDING_ALPHA = 110
TRAIL_POINT_SPACING_M = 6.0
MAX_TRAIL_POINTS = 1200
OSM_ZOOM_LEVEL = 16
TILE_CACHE_DIR = "cache"
COLLISION_RADIUS_M = 30.0
PAN_SPEED = 0.0008
if not os.path.exists(TILE_CACHE_DIR): os.makedirs(TILE_CACHE_DIR)

# ======== UTIL ========
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def node_latlon(G, n):
    return (G.nodes[n]['y'], G.nodes[n]['x'])

def deg2num(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    return (math.degrees(lat_rad), lon_deg)

# ======== TILE LOADER ========
class TileLoader:
    def __init__(self):
        self.cache = {}
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        while True:
            z, x, y = self.queue.get()
            path = os.path.join(TILE_CACHE_DIR, f"{z}_{x}_{y}.png")
            if not os.path.exists(path):
                try:
                    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                    headers = {'User-Agent': 'DroneSimVishnu/1.0'}
                    r = requests.get(url, timeout=10, headers=headers)
                    if r.status_code == 200:
                        with open(path, "wb") as f: f.write(r.content)
                except: pass
            try:
                surf = pygame.image.load(path).convert_alpha()
                self.cache[(z, x, y)] = surf
            except: pass
            self.queue.task_done()
    
    def get(self, z, x, y):
        key = (z, x, y)
        if key in self.cache: return self.cache[key]
        self.queue.put(key)
        return None

tile_loader = TileLoader()

# ======== WEATHER SYSTEM ========
class Weather:
    def __init__(self):
        self.raining = False
        self.cloud_alpha = 0
        self.rain_drops = []
        self.thunder_timer = 0
        self.lightning = False
        self.wind_gust = 0.0

    def update(self, dt):
        if random.random() < 0.0003:
            self.raining = not self.raining
            if not self.raining:
                self.rain_drops = []
        
        if self.raining:
            self.cloud_alpha = min(180, self.cloud_alpha + dt * 80)
            if len(self.rain_drops) < 800:
                for _ in range(25):
                    self.rain_drops.append([random.randint(-200, INIT_WIN_W + 200), random.randint(-100, 0), random.uniform(1.8, 3.5)])
            
            for drop in self.rain_drops[:]:
                drop[1] += drop[2] * 900 * dt
                if drop[1] > INIT_WIN_H + 100:
                    self.rain_drops.remove(drop)
            
            if random.random() < 0.0012:
                self.thunder_timer = 0.4
                self.lightning = True
        else:
            self.cloud_alpha = max(0, self.cloud_alpha - dt * 60)
        
        if self.thunder_timer > 0:
            self.thunder_timer -= dt
            if self.thunder_timer <= 0:
                self.lightning = False
        
        self.wind_gust = 5.0 if self.raining else 0.0

    def draw(self, screen):
        if self.cloud_alpha > 0:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((60, 70, 100, int(self.cloud_alpha)))
            screen.blit(overlay, (0, 0))
        
        for drop in self.rain_drops:
            x, y = int(drop[0]), int(drop[1])
            pygame.draw.line(screen, (200, 220, 255, 200), (x, y), (x + 2, y + 14), 2)
        
        if self.lightning:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 90))
            screen.blit(overlay, (0, 0))

weather = Weather()

# ======== WIND FIELD ========
class DynamicWindField:
    def __init__(self, bounds):
        self.zones = [(random.uniform(bounds[0], bounds[2]), random.uniform(bounds[1], bounds[3]), 
                       random.uniform(500, 1100), random.uniform(5, 17), 
                       random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)) for _ in range(6)]
        self.last_update = time.time()
    
    def update(self, dt):
        if time.time() - self.last_update < 7.0: return
        self.last_update = time.time()
        for i in range(len(self.zones)):
            c = list(self.zones[i])
            c[0] += c[4] * dt / 111000
            c[1] += c[5] * dt / 111000
            c[3] = max(3.0, c[3] + random.uniform(-2, 2))
            self.zones[i] = tuple(c)
    
    def get_wind(self, lat, lon):
        ms = 0.0
        for clat, clon, r, peak, _, _ in self.zones:
            d = haversine_m(lat, lon, clat, clon)
            if d <= r: ms = max(ms, peak * (1 - d/r))
        return ms + weather.wind_gust

# ======== A* PATHFINDING ========
_heap_counter = itertools.count()

def heuristic_25d(G25, a, b):
    Gosm = G25.graph['osm']
    la, lo = node_latlon(Gosm, a[0])
    lb, lob = node_latlon(Gosm, b[0])
    horiz = haversine_m(la, lo, lb, lob)
    vert = abs(a[1] - b[1]) * G25.graph['layer_h']
    return horiz + vert

def edge_cost_25d(G25, u, v, wind_field):
    Gosm = G25.graph['osm']
    ua, uz = u; vb, vz = v
    la, lo = node_latlon(Gosm, ua)
    lb, lob = node_latlon(Gosm, vb)
    horiz = haversine_m(la, lo, lb, lob)
    climb = max(0.0, (vz - uz) * G25.graph['layer_h'])
    midlat = (la + lb) / 2
    midlon = (lo + lob) / 2
    w = wind_field.get_wind(midlat, midlon) if wind_field else 0.0
    wind_pen = 100000.0 if w >= 10.0 else 1.0 + 0.05 * w
    return (horiz + climb) * wind_pen

def a_star_25d(G25, start_osm_node, goal_osm_node, wind_field):
    start = (start_osm_node, 0)
    goal_candidates = [(goal_osm_node, z) for z in range(LAYERS) if (goal_osm_node, z) in G25._node]
    if not goal_candidates: return None
    openh = []
    heapq.heappush(openh, (0.0, next(_heap_counter), start))
    gscore = {start: 0.0}
    parent = {}
    closed = set()
    while openh:
        f, _, cur = heapq.heappop(openh)
        if cur in closed: continue
        closed.add(cur)
        if cur in goal_candidates:
            path = []
            n = cur
            while n in parent:
                path.append(n)
                n = parent[n]
            path.append(start)
            path.reverse()
            return path
        for nbr in G25.neighbors(cur):
            if nbr not in G25._node: continue
            tentative = gscore[cur] + edge_cost_25d(G25, cur, nbr, wind_field)
            if tentative < gscore.get(nbr, float('inf')):
                gscore[nbr] = tentative
                parent[nbr] = cur
                h_vals = [heuristic_25d(G25, nbr, g) for g in goal_candidates]
                h = min(h_vals) if h_vals else 0
                heapq.heappush(openh, (tentative + h, next(_heap_counter), nbr))
    return None

# ======== DRONE CLASS ========
class Drone:
    def __init__(self, id, src_name, dst_name, path_states, coords, target_center, home):
        self.id = id; self.src_name = src_name; self.dst_name = dst_name
        self.path_states = path_states; self.coords = coords; self.idx = 0; self.subt = 0.0
        self.trail = []; self.returning = False; self.target_center = target_center; self.home = home
        self.total_distance_m = sum(haversine_m(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]) +
                                   abs(coords[i+1][2] - coords[i][2]) * LAYER_HEIGHT_M
                                   for i in range(len(coords)-1))
        self.emergency_climb = 0.0; self.collision_warning = 0.0

    def get_position(self):
        if self.idx >= len(self.coords) - 1:
            lat, lon, z = self.coords[-1]
        else:
            a, b = self.coords[self.idx], self.coords[self.idx+1]
            seg_time = max(0.2, haversine_m(a[0], a[1], b[0], b[1]) / DRONE_SPEED_MPS)
            frac = min(1.0, self.subt / seg_time)
            lat = a[0] + (b[0] - a[0]) * frac
            lon = a[1] + (b[1] - a[1]) * frac
            z = a[2] + (b[2] - a[2]) * frac
        alt = (z + self.emergency_climb) * LAYER_HEIGHT_M
        return lat, lon, alt

    def remaining_distance_m(self):
        rem = 0.0
        if self.idx < len(self.coords) - 1:
            a, b = self.coords[self.idx], self.coords[self.idx+1]
            horiz = haversine_m(a[0], a[1], b[0], b[1])
            seg_time = max(0.2, horiz / DRONE_SPEED_MPS)
            frac = self.subt / seg_time if seg_time > 0 else 0
            rem += (1.0 - frac) * horiz
            for i in range(self.idx + 1, len(self.coords) - 1):
                rem += haversine_m(*self.coords[i][:2], *self.coords[i+1][:2])
        return rem

    def eta_s(self):
        return max(1, int(self.remaining_distance_m() / (DRONE_SPEED_MPS * 0.85)))

    def record_trail_point(self, lat, lon):
        if not self.trail or haversine_m(self.trail[-1][0], self.trail[-1][1], lat, lon) >= TRAIL_POINT_SPACING_M:
            self.trail.append((lat, lon))
            if len(self.trail) > MAX_TRAIL_POINTS: self.trail.pop(0)

# ======== CITY GENERATOR ========
def create_synthetic_city():
    buildings = []
    np.random.seed(42)
    for i in range(55):
        angle = np.random.uniform(0, 2*math.pi)
        dist = np.random.uniform(250, 1750)
        lat = CENTER_LAT + (dist / 111000) * math.cos(angle)
        lon = CENTER_LON + (dist / 111000) * math.sin(angle) / math.cos(math.radians(CENTER_LAT))
        size = np.random.uniform(45, 110)
        height = np.random.uniform(25, 95)
        half = size / 2 / 111000
        ring = [(lon - half, lat - half), (lon + half, lat - half),
                (lon + half, lat + half), (lon - half, lat + half),
                (lon - half, lat - half)]
        buildings.append((i, ring, (lat, lon), height))
    return buildings

# ======== MAIN APP ========
class App:
    def __init__(self):
        pygame.init()
        self.WIN_W, self.WIN_H = INIT_WIN_W, INIT_WIN_H
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("Drone Delivery - FINAL PERFECTION")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        self.bigfont = pygame.font.SysFont("Consolas", 20, bold=True)
        self.smallfont = pygame.font.SysFont("Consolas", 14)
        self.G_osm = None; self.G25 = None; self.wind_field = None
        self.buildings = []; self.building_polygons = []; self.building_status = {}
        self.active_drones = {}; self.map_ready = False
        self.sources = {"Station A (Central)": (CENTER_LAT, CENTER_LON)}
        self.station_pool = {"Station A (Central)": {'available': 10, 'total': 10}}
        self.status = "Building world..."
        self.view_lat = CENTER_LAT; self.view_lon = CENTER_LON
        self.zoom_level = 1.0; self.last_tile_update = 0; self.collision_events = 0
        threading.Thread(target=self._load_map, daemon=True).start()

    def _load_map(self):
        self.buildings = create_synthetic_city()
        self.building_polygons = []
        for i, ring, center, height in self.buildings:
            poly = Polygon([(lon, lat) for lon, lat in ring])
            self.building_polygons.append((poly, height))
            self.building_status[f"Building {i}"] = None

        min_lat = min(c[0] for _, _, c, _ in self.buildings) - 0.004
        max_lat = max(c[0] for _, _, c, _ in self.buildings) + 0.004
        min_lon = min(c[1] for _, _, c, _ in self.buildings) - 0.004
        max_lon = max(c[1] for _, _, c, _ in self.buildings) + 0.004
        bounds = (min_lat, min_lon, max_lat, max_lon)

        lats = np.linspace(min_lat, max_lat, 70)
        lons = np.linspace(min_lon, max_lon, 70)
        self.G_osm = nx.Graph()
        for i in range(70):
            for j in range(70):
                nid = (i, j)
                self.G_osm.add_node(nid, y=lats[i], x=lons[j])
                if i > 0: self.G_osm.add_edge(nid, (i-1, j))
                if j > 0: self.G_osm.add_edge(nid, (i, j-1))

        G = nx.DiGraph()
        G.graph['osm'] = self.G_osm
        G.graph['layer_h'] = LAYER_HEIGHT_M
        remove = set()
        protected_nodes = set()
        for _, _, center, _ in self.buildings:
            n = self.find_nearest_grid_node(center[0], center[1])
            if n: protected_nodes.add((n, 0))

        for n in self.G_osm.nodes():
            lat, lon = node_latlon(self.G_osm, n)
            pt = Point(lon, lat)
            for z in range(LAYERS):
                node3d = (n, z)
                alt = z * LAYER_HEIGHT_M + 15
                inside = any(p.contains(pt) and alt < h for p, h in self.building_polygons)
                buffer_zone = any(p.buffer(25/111000).contains(pt) for p, _ in self.building_polygons)
                if (inside or buffer_zone) and node3d not in protected_nodes:
                    remove.add(node3d)
                else:
                    G.add_node(node3d)

        for u, v in self.G_osm.edges():
            for z in range(LAYERS):
                u3, v3 = (u, z), (v, z)
                if u3 in remove or v3 in remove: continue
                dist = haversine_m(*node_latlon(self.G_osm, u), *node_latlon(self.G_osm, v))
                G.add_edge(u3, v3, weight=dist)
                G.add_edge(v3, u3, weight=dist)
        for n in self.G_osm.nodes():
            for z in range(LAYERS-1):
                u, v = (n, z), (n, z+1)
                if u in remove or v in remove: continue
                G.add_edge(u, v, weight=LAYER_HEIGHT_M)
                G.add_edge(v, u, weight=0)

        G.remove_nodes_from(remove)
        self.G25 = G
        self.wind_field = DynamicWindField(bounds)
        self.map_ready = True
        self.status = "READY! Press SPACE to launch drones"

    def find_nearest_grid_node(self, lat, lon):
        best, best_d = None, float('inf')
        for n in self.G_osm.nodes():
            dlat, dlon = node_latlon(self.G_osm, n)
            d = haversine_m(lat, lon, dlat, dlon)
            if d < best_d:
                best_d, best = d, n
        return best if best_d < 800 else None

    def _proj(self, lat, lon):
        res = 156543.03 * math.cos(math.radians(self.view_lat)) / (2**OSM_ZOOM_LEVEL)
        dx = (lon - self.view_lon) * 111320 * math.cos(math.radians(self.view_lat))
        dy = (self.view_lat - lat) * 111000
        x = self.WIN_W // 2 + dx / res * self.zoom_level
        y = self.WIN_H // 2 + dy / res * self.zoom_level
        return int(x), int(y)

    def preload_tiles(self):
        now = time.time()
        if now - self.last_tile_update < 0.5: return
        self.last_tile_update = now
        cx, cy = deg2num(self.view_lat, self.view_lon, OSM_ZOOM_LEVEL)
        for i in range(-5, 6):
            for j in range(-5, 6):
                tile_loader.get(OSM_ZOOM_LEVEL, cx + i, cy + j)

    def draw_map_tiles(self):
        self.preload_tiles()
        cx, cy = deg2num(self.view_lat, self.view_lon, OSM_ZOOM_LEVEL)
        tile_size = int(256 * self.zoom_level)
        for i in range(-4, 5):
            for j in range(-4, 5):
                tile = tile_loader.cache.get((OSM_ZOOM_LEVEL, cx + i, cy + j))
                if tile:
                    lat, lon = num2deg(cx + i, cy + j, OSM_ZOOM_LEVEL)
                    x, y = self._proj(lat, lon)
                    scaled = pygame.transform.smoothscale(tile, (tile_size, tile_size))
                    self.screen.blit(scaled, (x - tile_size//2, y - tile_size//2))

    def draw_map(self):
        if not self.map_ready: return
        self.draw_map_tiles()
        surf = pygame.Surface((self.WIN_W, self.WIN_H), pygame.SRCALPHA)
        for i, ring, center, _ in self.buildings:
            color = (255, 50, 50, 200) if self.building_status.get(f"Building {i}") else (75, 75, 95, BUILDING_ALPHA)
            pts = [self._proj(lat, lon) for lon, lat in ring]
            if len(pts) > 2:
                pygame.draw.polygon(surf, color, pts)
            if self.building_status.get(f"Building {i}"):
                x, y = self._proj(*center)
                pygame.draw.circle(surf, (0, 255, 255), (x, y), 30)
                pygame.draw.circle(surf, (255, 255, 0), (x, y), 30, 10)
                pygame.draw.circle(surf, (255, 0, 255), (x, y), 18)
        self.screen.blit(surf, (0, 0))
        x, y = self._proj(*self.sources["Station A (Central)"])
        pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 30)
        pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 30, 8)

    def check_collisions(self, dt):
        drones = list(self.active_drones.values())
        for i, d1 in enumerate(drones):
            lat1, lon1, alt1 = d1.get_position()
            for d2 in drones[i+1:]:
                lat2, lon2, alt2 = d2.get_position()
                horiz = haversine_m(lat1, lon1, lat2, lon2)
                vert = abs(alt1 - alt2)
                dist = math.sqrt(horiz**2 + vert**2)
                if dist < COLLISION_RADIUS_M:
                    self.collision_events += 1
                    d1.collision_warning = 1.0; d2.collision_warning = 1.0
                    climb = (COLLISION_RADIUS_M - dist) / 2 / LAYER_HEIGHT_M
                    d1.emergency_climb += climb; d2.emergency_climb += climb
                    self.status = f"COLLISION AVOIDED #{self.collision_events}"

    def update_drones(self, dt):
        if not self.map_ready: return
        self.wind_field.update(dt)
        self.check_collisions(dt)
        done = []
        for drone in self.active_drones.values():
            drone.emergency_climb = max(0, drone.emergency_climb - dt * 0.5)
            drone.collision_warning = max(0, drone.collision_warning - dt)
            if drone.idx >= len(drone.coords) - 1:
                if not drone.returning:
                    drone.returning = True
                    ret_path, ret_coords, _ = self.plan_path(drone.target_center, drone.home)
                    if ret_coords:
                        drone.coords = ret_coords; drone.idx = 0; drone.subt = 0.0
                    else:
                        done.append(drone.id)
                else:
                    done.append(drone.id)
                continue
            # FIXED: was self.idx → now drone.idx
            a, b = drone.coords[drone.idx], drone.coords[drone.idx + 1]
            seg_time = max(0.2, haversine_m(a[0], a[1], b[0], b[1]) / DRONE_SPEED_MPS)
            drone.subt += dt
            if drone.subt >= seg_time:
                drone.idx += 1
                drone.subt = 0.0
            frac = min(1.0, drone.subt / seg_time)
            lat = a[0] + (b[0] - a[0]) * frac
            lon = a[1] + (b[1] - a[1]) * frac
            drone.record_trail_point(lat, lon)
        for did in done:
            d = self.active_drones[did]
            self.station_pool["Station A (Central)"]['available'] += 1
            self.building_status[d.dst_name] = None
            del self.active_drones[did]
            self.status = f"{did} delivered safely!"

    def draw_drones(self):
        for drone in self.active_drones.values():
            if len(drone.trail) > 1:
                pts = [self._proj(lat, lon) for lat, lon in drone.trail[-200:]]
                color = (255, 100, 100, 220) if drone.collision_warning > 0 else (80, 180, 255, 200)
                pygame.draw.lines(self.screen, color, False, pts, 4 if drone.collision_warning > 0 else 3)
            lat, lon, alt = drone.get_position()
            x, y = self._proj(lat, lon)
            size = 12 + int(alt / 10)
            color = (255, 0, 0) if drone.collision_warning > 0 else (255, 60, 60)
            pygame.draw.circle(self.screen, color, (x, y), size)
            if drone.collision_warning > 0:
                pygame.draw.circle(self.screen, (255, 255, 0), (x, y), size + 14, 6)

    def draw_hud(self):
        pygame.draw.rect(self.screen, (245, 245, 255, 240), (0, 0, self.WIN_W, 44))
        wind = f"{self.wind_field.get_wind(CENTER_LAT, CENTER_LON):.1f}" if self.wind_field else "0.0"
        rain = "RAINING" if weather.raining else "Clear"
        txt = f"Active: {len(self.active_drones)} | Avail: {self.station_pool['Station A (Central)']['available']}/10 | Wind: {wind}m/s | {rain} | Avoided: {self.collision_events} | SPACE=Launch"
        self.screen.blit(self.font.render(txt, True, (0, 0, 0)), (12, 12))
        pygame.draw.rect(self.screen, (245, 245, 255, 240), (0, self.WIN_H - 34, self.WIN_W, 34))
        color = (255, 0, 0) if "COLLISION" in self.status else (0, 0, 0)
        self.screen.blit(self.font.render(self.status, True, color), (12, self.WIN_H - 24))

        tx, ty = self.WIN_W - 430, 60
        pygame.draw.rect(self.screen, (255, 255, 255, 245), (tx, ty, 420, self.WIN_H - 120))
        self.screen.blit(self.bigfont.render("Active Drones", True, (0, 0, 0)), (tx + 15, ty + 10))
        headers = ["ID", "Total(km)", "Rem(km)", "ETA(s)"]
        cols = [tx + 20, tx + 110, tx + 220, tx + 320]
        y = ty + 50
        for i, h in enumerate(headers):
            self.screen.blit(self.smallfont.render(h, True, (0, 0, 0)), (cols[i], y))
        pygame.draw.line(self.screen, (0, 0, 0), (tx + 10, y + 20), (tx + 410, y + 20), 2)
        y += 30
        for drone in sorted(self.active_drones.values(), key=lambda d: d.id):
            if y > self.WIN_H - 50: break
            self.screen.blit(self.smallfont.render(drone.id, True, (0, 0, 180)), (cols[0], y))
            self.screen.blit(self.smallfont.render(f"{drone.total_distance_m/1000:.2f}", True, (60, 60, 60)), (cols[1], y))
            self.screen.blit(self.smallfont.render(f"{drone.remaining_distance_m()/1000:.2f}", True, (60, 60, 60)), (cols[2], y))
            self.screen.blit(self.smallfont.render(f"{drone.eta_s()}", True, (180, 0, 0)), (cols[3], y))
            y += 24

    def plan_path(self, src, dst):
        s = self.find_nearest_grid_node(src[0], src[1])
        g = self.find_nearest_grid_node(dst[0], dst[1])
        if not s or not g: return None, None, None
        path = a_star_25d(self.G25, s, g, self.wind_field)
        if not path: return None, None, None
        coords = [(src[0], src[1], 0)]
        for n, z in path:
            lat, lon = node_latlon(self.G_osm, n)
            coords.append((lat, lon, z))
        return path, coords, g

    def launch_drone(self):
        if not self.map_ready or self.station_pool["Station A (Central)"]['available'] <= 0:
            self.status = "No drones available!"
            return
        free = [b for b in self.buildings if not self.building_status.get(f"Building {b[0]}")]
        if not free:
            self.status = "All buildings busy!"
            return
        idx, _, center, _ = random.choice(free)
        path, coords, _ = self.plan_path(self.sources["Station A (Central)"], center)
        if not path:
            self.status = "No path found!"
            return
        drone_id = f"D{random.randint(100,999)}"
        self.station_pool["Station A (Central)"]['available'] -= 1
        self.building_status[f"Building {idx}"] = drone_id
        drone = Drone(drone_id, "Station A (Central)", f"Building {idx}", path, coords, center, self.sources["Station A (Central)"])
        self.active_drones[drone_id] = drone
        self.status = f"{drone_id} launched!"

    def run(self):
        while True:
            dt = self.clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT: return
                if e.type == pygame.VIDEORESIZE:
                    self.WIN_W, self.WIN_H = e.w, e.h
                    self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H), pygame.RESIZABLE)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE: self.launch_drone()
                    if e.key == pygame.K_c:
                        self.active_drones.clear()
                        self.station_pool["Station A (Central)"]['available'] = 10
                        for k in self.building_status: self.building_status[k] = None
                        self.status = "All drones recalled!"
                    if e.key in (pygame.K_PLUS, pygame.K_EQUALS): self.zoom_level *= 1.2
                    if e.key == pygame.K_MINUS: self.zoom_level /= 1.2

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: self.view_lat += PAN_SPEED / self.zoom_level
            if keys[pygame.K_DOWN]: self.view_lat -= PAN_SPEED / self.zoom_level
            if keys[pygame.K_LEFT]: self.view_lon -= PAN_SPEED / self.zoom_level
            if keys[pygame.K_RIGHT]: self.view_lon += PAN_SPEED / self.zoom_level

            self.screen.fill((135, 206, 250))
            if self.map_ready:
                weather.update(dt)
                self.update_drones(dt)
                self.draw_map()
                self.draw_drones()
                weather.draw(self.screen)
            self.draw_hud()
            pygame.display.flip()

# ----------------- BEGIN: Metrics helper (added) -----------------
# Small helper that re-uses the code's city/grid/A* logic and prints:
#  - Avg. delivery time (min)
#  - Success rate (%)
#  - Avg. energy (Wh)
# This does NOT change any simulation logic — only reads your existing functions/constants.

ENERGY_PER_M = 0.02  # Wh per meter (20 Wh / km). Change later if you prefer a different model.

_heap_counter_local = itertools.count()

def heuristic_25d_local(G25, a, b):
    Gosm = G25.graph['osm']
    la, lo = node_latlon(Gosm, a[0])
    lb, lob = node_latlon(Gosm, b[0])
    horiz = haversine_m(la, lo, lb, lob)
    vert = abs(a[1] - b[1]) * G25.graph['layer_h']
    return horiz + vert

def edge_cost_25d_local(G25, u, v):
    Gosm = G25.graph['osm']
    ua, uz = u; vb, vz = v
    la, lo = node_latlon(Gosm, ua)
    lb, lob = node_latlon(Gosm, vb)
    horiz = haversine_m(la, lo, lb, lob)
    climb = max(0.0, (vz - uz) * G25.graph['layer_h'])
    wind_pen = 1.0
    return (horiz + climb) * wind_pen

def a_star_25d_local(G25, start_osm_node, goal_osm_node):
    start = (start_osm_node, 0)
    goal_candidates = [(goal_osm_node, z) for z in range(LAYERS) if (goal_osm_node, z) in G25._node]
    if not goal_candidates:
        return None
    openh = []
    heapq.heappush(openh, (0.0, next(_heap_counter_local), start))
    gscore = {start: 0.0}
    parent = {}
    closed = set()
    while openh:
        f, _, cur = heapq.heappop(openh)
        if cur in closed: continue
        closed.add(cur)
        if cur in goal_candidates:
            path = []
            n = cur
            while n in parent:
                path.append(n)
                n = parent[n]
            path.append(start)
            path.reverse()
            return path
        for nbr in G25.neighbors(cur):
            if nbr not in G25._node: continue
            tentative = gscore[cur] + edge_cost_25d_local(G25, cur, nbr)
            if tentative < gscore.get(nbr, float('inf')):
                gscore[nbr] = tentative
                parent[nbr] = cur
                h_vals = [heuristic_25d_local(G25, nbr, g) for g in goal_candidates]
                h = min(h_vals) if h_vals else 0
                heapq.heappush(openh, (tentative + h, next(_heap_counter_local), nbr))
    return None

def compute_and_print_metrics():
    # Build same synthetic city and 70x70 grid as App._load_map
    buildings_local = create_synthetic_city()
    min_lat = min(c[0] for _, _, c, _ in buildings_local) - 0.004
    max_lat = max(c[0] for _, _, c, _ in buildings_local) + 0.004
    min_lon = min(c[1] for _, _, c, _ in buildings_local) - 0.004
    max_lon = max(c[1] for _, _, c, _ in buildings_local) + 0.004

    lats = np.linspace(min_lat, max_lat, 70)
    lons = np.linspace(min_lon, max_lon, 70)
    G_osm_local = nx.Graph()
    for i in range(70):
        for j in range(70):
            nid = (i, j)
            G_osm_local.add_node(nid, y=lats[i], x=lons[j])
            if i > 0: G_osm_local.add_edge(nid, (i-1, j))
            if j > 0: G_osm_local.add_edge(nid, (i, j-1))

    G_local = nx.DiGraph()
    G_local.graph['osm'] = G_osm_local
    G_local.graph['layer_h'] = LAYER_HEIGHT_M

    building_polys = [(Polygon([(lon, lat) for lon, lat in ring]), h) for _, ring, _, h in buildings_local]

    def find_nearest_grid_node_local(lat, lon):
        best, best_d = None, float('inf')
        for n in G_osm_local.nodes():
            dlat, dlon = node_latlon(G_osm_local, n)
            d = haversine_m(lat, lon, dlat, dlon)
            if d < best_d:
                best_d, best = d, n
        return best if best_d < 800 else None

    protected_nodes = set()
    for _, _, center, _ in buildings_local:
        n = find_nearest_grid_node_local(center[0], center[1])
        if n: protected_nodes.add((n, 0))

    remove = set()
    for n in G_osm_local.nodes():
        lat, lon = node_latlon(G_osm_local, n)
        pt = Point(lon, lat)
        for z in range(LAYERS):
            node3d = (n, z)
            alt = z * LAYER_HEIGHT_M + 15
            inside = any(p.contains(pt) and alt < h for p, h in building_polys)
            buffer_zone = any(p.buffer(25/111000).contains(pt) for p, _ in building_polys)
            if (inside or buffer_zone) and node3d not in protected_nodes:
                remove.add(node3d)
            else:
                G_local.add_node(node3d)

    for u, v in G_osm_local.edges():
        for z in range(LAYERS):
            u3, v3 = (u, z), (v, z)
            if u3 in remove or v3 in remove: continue
            dist = haversine_m(*node_latlon(G_osm_local, u), *node_latlon(G_osm_local, v))
            G_local.add_edge(u3, v3, weight=dist)
            G_local.add_edge(v3, u3, weight=dist)
    for n in G_osm_local.nodes():
        for z in range(LAYERS - 1):
            u, v = (n, z), (n, z + 1)
            if u in remove or v in remove: continue
            G_local.add_edge(u, v, weight=LAYER_HEIGHT_M)
            G_local.add_edge(v, u, weight=0)
    G_local.remove_nodes_from(remove)

    # compute metrics
    station = (CENTER_LAT, CENTER_LON)
    total = len(buildings_local)
    successful = 0
    delivery_times_s = []
    energies_wh = []

    s_node = find_nearest_grid_node_local(station[0], station[1])
    for i, ring, center, _ in buildings_local:
        g_node = find_nearest_grid_node_local(center[0], center[1])
        if not s_node or not g_node:
            continue
        path = a_star_25d_local(G_local, s_node, g_node)
        if not path:
            continue
        coords = [(station[0], station[1], 0)]
        for n, z in path:
            lat, lon = node_latlon(G_osm_local, n)
            coords.append((lat, lon, z))
        total_distance_m = 0.0
        for k in range(len(coords) - 1):
            total_distance_m += haversine_m(coords[k][0], coords[k][1], coords[k+1][0], coords[k+1][1]) + abs(coords[k+1][2] - coords[k][2]) * LAYER_HEIGHT_M
        eta_s = max(1, int(total_distance_m / (DRONE_SPEED_MPS * 0.85)))
        energy_wh = total_distance_m * ENERGY_PER_M
        successful += 1
        delivery_times_s.append(eta_s)
        energies_wh.append(energy_wh)

    success_rate = 100.0 * successful / total if total > 0 else 0.0
    avg_delivery_time_min = (sum(delivery_times_s) / len(delivery_times_s) / 60.0) if delivery_times_s else 0.0
    avg_energy_wh = (sum(energies_wh) / len(energies_wh)) if energies_wh else 0.0

    # Print in the same table layout as your screenshot (Baseline=Proposed since only one scenario present)
    print("\nTABLE I: Example results summary (from current code)\n")
    print(f"{'Metric':40s} {'Baseline':>10s} {'Proposed':>10s} {'Improvement':>12s}")
    print("-" * 74)
    print(f"{'Avg. delivery time (min)':40s} {avg_delivery_time_min:10.2f} {avg_delivery_time_min:10.2f} {0.0:12.2f}")
    print(f"{'Success rate (%)':40s} {success_rate:10.2f} {success_rate:10.2f} {0.0:12.2f}")
    print(f"{'Avg. energy (Wh)':40s} {avg_energy_wh:10.2f} {avg_energy_wh:10.2f} {0.0:12.2f}")
    print("\n(Notes: ENERGY_PER_M = {:.4f} Wh/m used for energy calc.)\n".format(ENERGY_PER_M))
# ----------------- END: Metrics helper -----------------

if __name__ == "__main__":
    # compute & print metrics derived from current code logic, then start the interactive app
    compute_and_print_metrics()
    App().run()
