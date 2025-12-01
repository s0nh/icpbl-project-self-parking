"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import heapq
from collections import deque

def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)
    stationary = map_payload.get("grid", {}).get("stationary")
    if stationary:
        rows = len(stationary)
        cols = len(stationary[0]) if stationary else 0
        print("[algo] grid size  :", rows, "x", cols)


@dataclass
class PlannerSkeleton:
    """경로 계획/제어 로직을 담는 기본 스켈레톤 클래스입니다."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    waypoints: List[Tuple[float, float]] = None
    base_board = None
    collision_map = None
    cur_idx = 0
    target_yaw_dir = "up"
    is_reverse = False

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터에서 전송한 정적 맵 데이터를 보관합니다."""

        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")
        self.parked_grid = map_payload.get("grid", {}).get("parked")
        self.slots = map_payload.get("slots")
        self.occupied_idx = map_payload.get("occupied_idx")
        self.walls = map_payload.get("walls_rects")
        self.lines = map_payload.get("lines")
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()
        self.hmap = None
        self.collision_map = None
        self.expected_orientation = map_payload["expected_orientation"]

        self.base_board = [[0.0]*len(self.parked_grid[0]) for _ in range(len(self.parked_grid))]
        self.collision_map = [[0.0]*len(self.parked_grid[0]) for _ in range(len(self.parked_grid))]
        q = deque()

        def transform_idx(x_min, x_max, y_min, y_max):
            x_min -= self.map_extent[0]
            x_max -= self.map_extent[0]
            y_max, y_min = self.map_extent[3]-y_min, self.map_extent[3]-y_max
            x_min /= self.cell_size
            x_max /= self.cell_size
            y_min /= self.cell_size
            y_max /= self.cell_size
            return map(round, (x_min, x_max, y_min, y_max))
        
        ## occupied_slots
        for i in range(len(self.slots)):
            if self.occupied_idx[i]:
                x_min, x_max, y_min, y_max = self.slots[i]
                x_min, x_max, y_min, y_max = transform_idx(x_min, x_max, y_min, y_max)
                for i in range(y_min-1, y_max):
                    for j in range(x_min, x_max+1):
                        if i < 0 or j >= len(self.base_board[0]):
                            continue
                        self.base_board[i][j] = 1.0
                        self.collision_map[i][j] = 1.0
                        q.append((j, i)) ## 장애물 좌표 (x, y)
        
        ## walls
        for x_min, x_max, y_min, y_max in self.walls:
            x_min, x_max, y_min, y_max = transform_idx(x_min, x_max, y_min, y_max)
            for i in range(y_min-1, y_max):
                for j in range(x_min, x_max+1):
                    if i < 0 or j >= len(self.base_board[0]):
                        continue
                    self.base_board[i][j] = 1.0
                    self.collision_map[i][j] = 1.0
                    q.append((j, i)) ## 장애물 좌표 (x, y)
        
        ## lines
        for x_min, y_min, x_max, y_max in self.lines:
            x_min, x_max, y_min, y_max = transform_idx(x_min, x_max, y_min, y_max)
            for i in range(y_min-1, y_max):
                for j in range(x_min, x_max+1):
                    if i < 0 or j >= len(self.base_board[0]):
                        continue
                    self.base_board[i][j] = 1.0
                    self.collision_map[i][j] = 1.0
                    q.append((j, i)) ## 장애물 좌표 (x, y)
        
        ## 최종 base_board 확인용
        # with open("total_map.txt", "w", encoding="utf-8") as f:
        #     for p in self.base_board:
        #         print(*list(map(int, p)), file=f)

        while q:
            cx, cy = q.popleft()
            current_dist = self.collision_map[cy][cx]

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < len(self.base_board[0]) and 0 <= ny < len(self.base_board):
                    if self.collision_map[ny][nx] == 0:
                        self.collision_map[ny][nx] = current_dist + 1
                        if self.collision_map[ny][nx] == 5:
                            continue
                        q.append((nx, ny))


        # with open("collision_map.txt", "w", encoding="utf-8") as f:
        #     for p in self.collision_map:
        #         print(*list(map(int, p)), file=f)


























## ==============================================          compute_path             ==============================================================================================
    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""

        # TODO: A*, RRT*, Hybrid A* 등으로 self.waypoints를 채우세요.
        self.waypoints.clear()
            
        def to_idx(x, y):
            return round((x - self.map_extent[0])/self.cell_size), round((self.map_extent[3] - y)/self.cell_size)
        
        ## 관측값 저장
        x = obs["state"]["x"]
        y = obs["state"]["y"]
        yaw = obs["state"]["yaw"]
        v = obs["state"]["v"]
        x1, x2, y1, y2 = obs["target_slot"]
        dt = obs["limits"]["dt"]
        L = obs["limits"]["L"]
        maxSteer = obs["limits"]["maxSteer"]
        maxAccel = obs["limits"]["maxAccel"]
        maxBrake = obs["limits"]["maxBrake"]
        steerRate = obs["limits"]["steerRate"]

        self.target_yaw_dir = "up"
        target_middle_x_idx, target_middle_y_idx = to_idx((x1+x2)/2, (y1+y2)/2)
        if self.expected_orientation == "front_in" :
            if target_middle_y_idx > len(self.base_board)/3:
                self.target_yaw_dir = "down"
        else:
            if target_middle_y_idx < len(self.base_board)/3:
                self.target_yaw_dir = "down"
        
        target_x = (x1 + x2) / 2
        target_y = -1
        if self.target_yaw_dir == "up": # 후륜지점으로 변경.
            target_y = (y1 + y2) / 2 - L/6 # TODO: 왜 L/2가 아닐까? 고려
        else:
            target_y = (y1 + y2) / 2 + L/6

        target_hmap = [[-1] * len(self.base_board[0]) for _ in range(len(self.base_board))] ## target_pnt 로부터의 BFS 거리
        target_x_idx, target_y_idx = to_idx(target_x, target_y)
        self.collision_map[target_y_idx-1][target_x_idx] = 0
        self.collision_map[target_y_idx+1][target_x_idx] = 0
        target_hmap[target_y_idx][target_x_idx] = 0
        q = deque([(target_x_idx, target_y_idx)])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < len(self.base_board[0]) and 0 <= ny < len(self.base_board):
                    if target_hmap[ny][nx] == -1 and self.collision_map[ny][nx] == 0:
                        target_hmap[ny][nx] = target_hmap[cy][cx] + self.cell_size
                        q.append((nx, ny))

        ##-------------------------DEBUG---------------------------
        # with open("target_hmap.txt", "w", encoding="utf-8") as f:
        #     for p in target_hmap:
        #         print(*p, sep='\t', file=f)
        ##---------------------------------------------------------

        start_hmap = [[-1] * len(self.base_board[0]) for _ in range(len(self.base_board))] ## start_pnt 로부터의 BFS 거리
        start_x, start_y = x, y
        start_x_idx, start_y_idx = to_idx(x, y)
        start_hmap[start_y_idx][start_x_idx] = 0
        q = deque([(start_x_idx, start_y_idx)])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < len(self.base_board[0]) and 0 <= ny < len(self.base_board):
                    if start_hmap[ny][nx] == -1 and self.collision_map[ny][nx] == 0:
                        start_hmap[ny][nx] = start_hmap[cy][cx] + self.cell_size
                        q.append((nx, ny))

        ##-------------------------DEBUG---------------------------
        # with open("start_hmap.txt", "w", encoding="utf-8") as f:
        #     for p in start_hmap:
        #         print(*p, sep='\t', file=f)
        ##---------------------------------------------------------

        class Node:
            __slots__ = ("x", "y", "yaw", "x_idx", "y_idx", "yaw_idx", "steer", "direction", "cost", "kind", "parent")
            def __init__(self, x, y, yaw, x_idx, y_idx, yaw_idx, steer, direction, cost, kind, parent):
                self.x = x
                self.y = y
                self.yaw = yaw
                self.x_idx = x_idx
                self.y_idx = y_idx
                self.yaw_idx = yaw_idx
                self.steer = steer
                self.direction = direction
                self.cost = cost
                self.kind = kind ## 시작점부터 시작한 노드 or 목적지부터 시작한 노드
                self.parent = parent

            def __lt__(self, other):
                return self.cost < other.cost

        start_yaw_idx = round(yaw / 0.26)
        start_node = Node(start_x, start_y, yaw, start_x_idx, start_y_idx, start_yaw_idx, 0.0, 1, 0.0, "start", None)

        target_yaw = yaw if self.target_yaw_dir == "up" else -yaw
        target_yaw_idx = round(yaw / 0.26)
        target_node = Node(target_x, target_y, target_yaw, target_x_idx, target_y_idx, target_yaw_idx, 0.0, 1, 0.0, "target", None)
        if self.expected_orientation == "front_in":
            target_node.direction = -1
        pq = [(target_hmap[start_y_idx][start_x_idx], start_node), (start_hmap[target_y_idx][target_x_idx], target_node)]
        
        visited = {}
        visited[(start_x_idx, start_y_idx, start_yaw_idx)] = start_node
        visited[(target_x_idx, target_y_idx, target_yaw_idx)] = target_node

        while pq:
            f, node = heapq.heappop(pq)
            steer_inputs = [-maxSteer, -maxSteer/2, 0, maxSteer/2, maxSteer]
            for next_steer in steer_inputs:
                if abs(next_steer-node.steer) > maxSteer:
                    continue
                next_direction = node.direction
                distance = next_direction * self.cell_size * 1.5

                if abs(next_steer) < 0.001: ## 직진
                    next_yaw = node.yaw
                    next_x = node.x + distance * math.cos(node.yaw)
                    next_y = node.y + distance * math.sin(node.yaw)
                else: ## 회전
                    beta = (distance / L) * math.tan(next_steer)
                    R = L / math.tan(next_steer)
                    next_yaw = node.yaw + beta
                    next_x = node.x + R * (math.sin(next_yaw) - math.sin(node.yaw))
                    next_y = node.y - R * (math.cos(next_yaw) - math.cos(node.yaw))
                    # 각도 정규화 (-pi ~ pi)
                    next_yaw = (next_yaw + math.pi) % (2 * math.pi) - math.pi

                next_yaw_idx = round(next_yaw/0.26)
                next_x_idx, next_y_idx = to_idx(next_x, next_y)
                if not (0 <= next_x_idx < len(self.base_board[0]) and 0 <= next_y_idx < len(self.base_board)):
                    continue

                ## 이미 지나간 경로인지 확인
                if (next_x_idx, next_y_idx, next_yaw_idx) in visited:
                    ## 시작점에서 시작한 노드와 목적지에서 시작한 노드가 만나면 종료
                    if visited[(next_x_idx, next_y_idx, next_yaw_idx)].kind != node.kind:
                        first_node, final_node = visited[(next_x_idx, next_y_idx, next_yaw_idx)], node
                        if node.kind == "start":
                            first_node, final_node = node, visited[(next_x_idx, next_y_idx, next_yaw_idx)]
                        path = []
                        while first_node:
                            path.append((first_node.x, first_node.y))
                            first_node = first_node.parent
                        path = path[::-1]
                        while final_node:
                            path.append((final_node.x, final_node.y))
                            final_node = final_node.parent
                        self.waypoints = path
                        
                        # control 함수 돌리기 전 전처리
                        self.cur_idx = 0
                        self.is_reverse = False

                        ##-------------------------DEBUG---------------------------
                        debug_map = []
                        for p in self.base_board:
                            debug_map.append(p[:])
                        print(path)
                        for x, y in path:
                            x_idx, y_idx = to_idx(x, y)
                            if 0 <= x_idx < len(self.base_board[0]) and 0 <= y_idx < len(self.base_board):
                                debug_map[y_idx][x_idx] = 1
                        with open("final_path.txt", "w", encoding="utf-8") as f:
                            for p in debug_map:
                                print(*p, sep='\t', file=f)
                        ##---------------------------------------------------------
                        return
                    continue

                new_h = 999
                if node.kind == "start":
                    if target_hmap[next_y_idx][next_x_idx] < 0:
                        continue
                    new_h = target_hmap[next_y_idx][next_x_idx]
                else:
                    if start_hmap[next_y_idx][next_x_idx] < 0:
                        continue
                    new_h = start_hmap[next_y_idx][next_x_idx]
                

                step_cost = abs(distance)
                
                if abs(next_steer - node.steer) > 0.001:
                    step_cost += abs(distance)*2 ## 핸들조작 패널티

                new_cost = node.cost + step_cost
                next_node = Node(next_x, next_y, next_yaw, next_x_idx, next_y_idx, next_yaw_idx, next_steer, next_direction, new_cost, node.kind, node)

                visited[(next_x_idx, next_y_idx, next_yaw_idx)] = next_node

                heapq.heappush(pq, (new_h+new_cost, next_node))
    
## =========================================================================================================================================================================


























    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""
        if not self.waypoints:
            self.compute_path(obs)
            if not self.waypoints:
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # OBS 불러오기
        v = float(obs.get("state", {}).get("v", 0.0))
        x = obs["state"]["x"]
        y = obs["state"]["y"]
        yaw = obs["state"]["yaw"] 
        orientation = obs["state"].get("expected_orientation", None)
        #target_x1, target_x2, target_y1, target_y2 = obs["target_slot"]
        #target_x, target_y = (target_x1 + target_x2) / 2, (target_y1 + target_y2)/2
        dt = obs["limits"]["dt"]
        L = obs["limits"]["L"]
        maxSteer = obs["limits"]["maxSteer"]
        maxAccel = obs["limits"]["maxAccel"]
        maxBrake = obs["limits"]["maxBrake"]
        steerRate = obs["limits"]["steerRate"]

        # 조향각 구하기 Stanley 알고리즘 사용

        target_x, target_y = self.waypoints[-1] # 후륜 지점.
        
        # 목표지점 전륜 좌표.
        target_x_front = target_x
        if self.target_yaw_dir == "up":
                target_y_front = target_y + L
        else:
                target_y_front = target_y - L

        # 현재 좌표 전륜 위치 계산
        front_x = x + L * math.cos(yaw)
        front_y = y + L * math.sin(yaw)

        search_x, search_y = x, y
        if not self.is_reverse:
            search_x = x + L * math.cos(yaw)
            search_y = y + L * math.sin(yaw)
            
        # 기어번경 지점 탐색
        switch_idx = -1
        dist_to_switch = float('inf')
        search_limit = min(self.cur_idx + 20, len(self.waypoints) - 1)

        for i in range(self.cur_idx, search_limit):
            if i + 2 < len(self.waypoints):
                curr_p = self.waypoints[i]
                next_p = self.waypoints[i+1]
                yaw1 = math.atan2(next_p[1] - curr_p[1], next_p[0] - curr_p[0])
                next_next_p = self.waypoints[i+2]
                yaw2 = math.atan2(next_next_p[1] - next_p[1], next_next_p[0] - next_p[0])
                diff = abs(yaw1 - yaw2)
                diff = (diff + math.pi) % (2 * math.pi) - math.pi

                if abs(diff) > math.pi / 2: # math.radians(150)
                    switch_idx = i+1
                    dist_to_switch = math.hypot(x - self.waypoints[switch_idx][0], y - self.waypoints[switch_idx][1])
                    break

        # 가까운 경로 찾기
        min_dist = float('inf')
        start_search = self.cur_idx
        end_search = min(len(self.waypoints), self.cur_idx + 20)

        if switch_idx != -1 and dist_to_switch > 0.5 * self.cell_size:
            end_search = min(end_search, switch_idx + 1)
        for i in range(start_search, end_search):
            wx, wy = self.waypoints[i]
            wx, wy = wx + L*math.cos(yaw), wy + L*math.sin(yaw)
            dist = math.hypot(search_x - wx, search_y - wy)
            if dist < min_dist:
                min_dist = dist
                self.cur_idx = i

        # 조향각 계산
        tx, ty = self.waypoints[self.cur_idx]

        if self.cur_idx + 1 < len(self.waypoints):
            nx, ny = self.waypoints[self.cur_idx + 1]
            path_yaw = math.atan2(ny - ty, nx - tx)
        else:
            nx, ny = tx, ty
            px, py = self.waypoints[self.cur_idx - 1]
            path_yaw = math.atan2(ty - py, tx - px)

        angle_to_path = path_yaw - yaw
        angle_to_path = (angle_to_path + math.pi) % (2 * math.pi) - math.pi

        # 후진 조향각 처리
        if not self.is_reverse:
            self.is_reverse = abs(angle_to_path) > math.pi / 2

        dist_to_goal = math.hypot(target_x_front - front_x, target_y_front - front_y)
              
        if self.is_reverse:
            cte = -math.sin(path_yaw) * (x- tx) + \
                math.cos(path_yaw) * (x - ty)
            
            heading_error = (path_yaw - yaw + math.pi) % (2*math.pi) - math.pi
            
            if dist_to_goal > 5.0 * self.cell_size:
                k = 1.0
                soft_v = 3.0
            else:
                k = 0.8
                soft_v = 3.0 
            
            new_steer = heading_error - math.atan2(k * cte, max(abs(v), soft_v))
            
        else:
            target_front_x = tx + L * math.cos(path_yaw)
            target_front_y = ty + L * math.sin(path_yaw)
            
            cte = -math.sin(path_yaw) * (front_x - target_front_x) + \
                math.cos(path_yaw) * (front_y - target_front_y)
            # 전진 파라미터
            k = 4.0
            soft_v = 0.1  
            new_steer = angle_to_path - math.atan2(k * cte, max(abs(v), soft_v))

        new_steer = new_steer if maxSteer > abs(new_steer) else maxSteer if new_steer > 0 else -maxSteer

        # 속도 제어 (P 제어기)

        check_dist = dist_to_goal

        if switch_idx != -1:
            check_dist = dist_to_switch

        if check_dist > 15.0*self.cell_size:
            target_v = 8.0*self.cell_size
        elif check_dist > 10.0*self.cell_size:
            target_v = 6.0*self.cell_size
        elif check_dist > 5.0*self.cell_size:
            target_v = 3.0*self.cell_size
        elif check_dist > 1.5*self.cell_size:
            target_v = 1.5*self.cell_size
        elif check_dist > 0.5*self.cell_size:
            target_v = 0.25*self.cell_size
        else:
            target_v = 0.0

        if switch_idx != -1:
            if dist_to_switch > 0.2 * self.cell_size:
                    target_v = max(target_v, 1.0*self.cell_size)

        current_v = v
        speed_error = target_v - abs(current_v)

        Kp = 1.0  # 튜닝 상수
        cmd = Kp * speed_error

        accel = 0.0
        brake = 0.0

        if cmd > 0:
            accel = min(cmd, maxAccel)
            brake = 0.0
        else:
            brake = -cmd
            accel = 0.0
            brake = min(brake, maxBrake)

        if self.is_reverse:
            if v > 0.1:
                accel = 0.0
                brake = maxBrake
        else:
            if v < -0.1:
                accel = 0.0
                brake = maxBrake

        if self.is_reverse:
            if dist_to_goal < 2.0*self.cell_size:
                accel = 0.0
                brake = maxBrake
                
                new_steer = 0.0
        else:
            if dist_to_goal < 0.3*self.cell_size:
                accel = 0.0
                brake = maxBrake

        print(f"cellsize: {self.cell_size}, dist_to_goal: {dist_to_goal:.2f}, target_v: {target_v:.2f}, steer: {new_steer:.2f}, accel: {accel:.2f}, brake: {brake:.2f}, target_point: {target_x_front:.2f}, {target_y_front:.2f}")
        return {
            "steer": float(new_steer),
            "accel": float(accel),
            "brake": float(brake),
            "gear": "R" if self.is_reverse else "D"
        }




        # 예시: 기본 데모 제어. 학생은 원하는 알고리즘으로 대체하면 됩니다.

        """
        t = float(obs.get("t", 0.0))
        v = float(obs.get("state", {}).get("v", 0.0))
        x = obs["state"]["x"] - self.map_extent[0] ## 0 ~ max_x로 범위 제한
        y = self.map_extent[3] - obs["state"]["y"] ## 0 ~ max_y로 범위 제한
        print(x, y)
        yaw = obs["state"]["yaw"] 
        target_x1, target_x2, target_y1, target_y2 = obs["target_slot"]
        target_x, target_y = (target_x1 + target_x2) / 2 - self.map_extent[0], self.map_extent[3] - (target_y1 + target_y2)/2
        dt = obs["limits"]["dt"]
        L = obs["limits"]["L"]
        maxSteer = obs["limits"]["maxSteer"]
        maxAccel = obs["limits"]["maxAccel"]
        maxBrake = obs["limits"]["maxBrake"]
        steerRate = obs["limits"]["steerRate"]
        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        min_dist = float('inf')
        current_idx = -1

        for i, (wx, wy) in enumerate(self.waypoints):
            if math.hypot(x-wx, y-wy) < min_dist:
                min_dist = math.hypot(x-wx, y-wy)
                current_idx = i
        
        lookahead_dist = 1.5 * self.cell_size * v
        target_point = self.waypoints[-1][0], self.map_extent[3]-self.waypoints[-1][1]

        for i in range(current_idx, len(self.waypoints)):
            wx, wy = self.waypoints[i][0], self.map_extent[3]-self.waypoints[i][1]
            dist = math.hypot(wx - x, wy - y)
            if dist > lookahead_dist:
                target_point = (wx, wy)
                break
        
        tx, ty = target_point
        print(tx, ty)
        angle_to_target = math.atan2(ty - y, tx - x)
        alpha = angle_to_target - yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        gear = "D"
        if abs(alpha) > math.pi/2: ## 차량 후방 -> 후진해야 하는 상황
            gear = "R"
            alpha = yaw+math.pi-alpha
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        steer_cmd = math.atan2(2.0 * L * math.sin(alpha), lookahead_dist)

        final_x, final_y = self.waypoints[-1]
        dist_to_goal = math.hypot(final_x - x, final_y - y)

        accel_cmd = 0.0
        brake_cmd = 0.0

        # 목표 속도 설정
        if dist_to_goal < 0.5:   # 도착 완료 (0.5m 이내)
            target_v = 0.0
            brake_cmd = 1.0      # 풀 브레이크
        elif dist_to_goal < 3.0: # 근처 진입
            target_v = 1.0       # 서행
        else:
            target_v = 2.5       # 주행
        
        # 이미 도착했으면 조향도 멈춤
        if dist_to_goal < 0.5:
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": gear}

        # 속도 오차에 따른 제어 (후진일 때는 속도 부호 고려)
        current_speed_abs = abs(v)
        if current_speed_abs < target_v:
            accel_cmd = 0.5 # 가속 (단순화: 목표보다 느리면 밟음)
            brake_cmd = 0.0
        else:
            accel_cmd = 0.0
            brake_cmd = 0.2 # 감속

        
        return {
            "steer": float(steer_cmd), 
            "accel": float(accel_cmd), 
            "brake": float(brake_cmd), 
            "gear": gear
        }"""





























# 전역 planner 인스턴스 (통신 모듈이 이 객체를 사용합니다.)
planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    """통신 모듈에서 맵 패킷을 받을 때 호출됩니다."""

    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """통신 모듈에서 매 스텝 호출하여 명령을 생성합니다."""

    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}

