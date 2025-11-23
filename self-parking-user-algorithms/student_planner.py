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
    hmap = None
    collision_map = None
    cur_idx = 0

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

        self.base_board = [[0.0]*len(self.parked_grid[0]) for _ in range(len(self.parked_grid))]
        self.collision_map = [[999.0]*len(self.parked_grid[0]) for _ in range(len(self.parked_grid))]
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
        with open("total_map.txt", "w", encoding="utf-8") as f:
            for p in self.base_board:
                print(*list(map(int, p)), file=f)

        while q:
            cx, cy = q.popleft()
            current_dist = self.collision_map[cy][cx]

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < len(self.base_board[0]) and 0 <= ny < len(self.base_board):
                    if self.collision_map[ny][nx] > current_dist + self.cell_size:
                        self.collision_map[ny][nx] = current_dist + self.cell_size
                        q.append((nx, ny))


        with open("collision_map.txt", "w", encoding="utf-8") as f:
            for p in self.collision_map:
                print(*list(map(int, p)), file=f)































## ==============================================          compute_path             ==============================================================================================
    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""

        # TODO: A*, RRT*, Hybrid A* 등으로 self.waypoints를 채우세요.
        self.waypoints.clear()
        self.cur_idx = 0
        ## 충돌여부 확인 함수 구현 - 정확한 차의 너비와 길이를 모르는 상태임;;
        def check_collision(x, y, yaw, width, height):
            
            front_len = height * 0.75
            rear_len = height * 0.25
            half_width = width / 2

            margin = 0.3
            f = front_len + margin
            r = rear_len + margin
            w = half_width + margin

            check_pnts = [
                (f, w),
                (f, -w),
                (-r, w),
                (-r, -w),
                (f, 0),
                (-r, 0),
                (0, w),
                (0, -w),
            ]

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            for lx, ly in check_pnts:
                gx = x + (lx * cos_yaw - ly * sin_yaw)
                gy = y + (lx * sin_yaw + ly * cos_yaw)
                col, row = round(gx/self.cell_size), round(gy/self.cell_size)
                if not (0 <= col < len(self.base_board[0]) and 0 <= row < len(self.base_board)):
                    return True
                if self.base_board[row][col] == 1:
                    return True
                
            return False
            """
            safe_radius = math.hypot(width/2, height/2)
            check_pnts = [
                (width/2, height/2),
                (width/2, -height/2),
                (-width/2, height/2),
                (-width/2, -height/2),
                # (width/2, height/4),
                # (width/2, -height/4),
                # (-width/2, height/4),
                # (-width/2, -height/4),
                (width/2, 0),
                (-width/2, 0),
                (0, height/2),
                (0, -height/2),
            ]
            x_idx, y_idx = round(x / self.cell_size), round(y / self.cell_size)
            if not (0 <= x_idx < len(self.base_board[0]) and 0 <= y_idx < len(self.base_board)):
                return True
            if self.collision_map[y_idx][x_idx] > safe_radius:
                return False
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            for lx, ly in check_pnts:
                gx = x + (lx * cos_yaw - ly * sin_yaw)
                gy = y + (lx * sin_yaw + ly * cos_yaw)
                col, row = round(gx/self.cell_size), round(gy/self.cell_size)
                if 0 <= col < len(self.base_board[0]) and 0 <= row < len(self.base_board):
                    if self.base_board[row][col] == 1:
                        return True
                else:
                    return True
            return False
            """

        ## 관측값 저장
        x = obs["state"]["x"] - self.map_extent[0] ## 0 ~ max_x로 범위 제한
        y = self.map_extent[3] - obs["state"]["y"] ## 0 ~ max_y로 범위 제한
        yaw = -obs["state"]["yaw"] 
        v = obs["state"]["v"]
        target_x1, target_x2, target_y1, target_y2 = obs["target_slot"]
        target_x, target_y = (target_x1 + target_x2) / 2 - self.map_extent[0], self.map_extent[3] - (target_y1 + target_y2)/2
        dt = obs["limits"]["dt"]
        L = obs["limits"]["L"]
        maxSteer = obs["limits"]["maxSteer"]
        maxAccel = obs["limits"]["maxAccel"]
        maxBrake = obs["limits"]["maxBrake"]
        steerRate = obs["limits"]["steerRate"]
        
        ## 휴리스틱 계산용 hmap 생성
        target_x_idx = round(target_x / self.cell_size)
        target_y_idx = round(target_y / self.cell_size)
        if self.hmap is None:
            hmap = [[-1]*len(self.base_board[0]) for _ in range(len(self.base_board))]
            q = deque([(target_x_idx, target_y_idx)])
            hmap[target_y_idx][target_x_idx] = 0
            while q:
                current_x, current_y = q.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= current_x+dx < len(self.base_board[1]) and 0 <= current_y+dy < len(self.base_board):
                        if self.base_board[current_y+dy][current_x+dx] == 0 and hmap[current_y+dy][current_x+dx] == -1:
                            hmap[current_y+dy][current_x+dx] = hmap[current_y][current_x]+self.cell_size
                            q.append((current_x+dx, current_y+dy))
            self.hmap = hmap
            with open("hmap.txt", "w", encoding="utf-8") as f:
                for p in self.hmap:
                    print(*p, sep='\t', file=f)
        
        ## 실세계 물리연산 실행할 최소 단위 Node 설정
        class Node:
            __slots__ = ("x", "y", "yaw", "x_idx", "y_idx", "yaw_idx", "steer", "v", "cost", "parent")
            def __init__(self, x, y, yaw, x_idx, y_idx, yaw_idx, steer, v, cost, parent):
                self.x = x
                self.y = y
                self.yaw = yaw
                self.x_idx = x_idx
                self.y_idx = y_idx
                self.yaw_idx = yaw_idx
                self.steer = steer
                self.v = v
                self.cost = cost
                self.parent = parent
            
            def __lt__(self, other):
                return self.cost < other.cost
        

        ## 연속형 좌표 -> 이산형 좌표 변환
        x_idx = round(x / self.cell_size)
        y_idx = round(y / self.cell_size)
        yaw_idx = round(yaw / 0.26) ## 15도로 나눠서 전방향 24개로 구분

        ## hybrid A* 구현
        start_node = Node(x, y, yaw, x_idx, y_idx, yaw_idx, 0.0, v, 0, None)

        pq = [(self.hmap[y_idx][x_idx], 0, start_node)]

        visited = set()
        visited.add((x_idx, y_idx, yaw_idx))
        
        while pq:
            h, g, node = heapq.heappop(pq)
            # 목표 도달 체크 (거리 0.5m 이내)
            if math.hypot(node.x - target_x, node.y - target_y) < 1.0:
                path = []
                curr = node
                
                ## ============================ 디버깅용
                print("found")
                debug_map = []
                for p in self.base_board:
                    debug_map.append(p.copy())
                ## ============================ 디버깅용
                # 부모 노드를 따라 시작점까지 거슬러 올라감
                while curr is not None:
                    debug_map[curr.y_idx][curr.x_idx] = 1.0
                    path.append((curr.x, curr.y))
                    curr = curr.parent
                with open("debug.txt", "w", encoding="utf-8") as f:
                    for p in debug_map:
                        print(*p, file=f)
                self.waypoints = path[::-1]
                break

            steer_inputs = [-maxSteer, -maxSteer*2/3, -maxSteer/3, 0, maxSteer/3, maxSteer*2/3, maxSteer]
            for next_steer in steer_inputs: ## 핸들 조작 적용
                if abs(next_steer-node.steer) > maxSteer*1.5:
                    continue

                for next_v in [-1, 1]: ## 가감속 적용
                    
                    distance = next_v
                    
                    ## 가능한 물리 엔진 적용
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
                    next_x_idx = round(next_x / self.cell_size)
                    next_y_idx = round(next_y / self.cell_size)

                    ## 충돌 확인
                    if check_collision(next_x, next_y, next_yaw, L*1.4, L*1.8):
                        continue
                    
                    if self.hmap[next_y_idx][next_x_idx] == -1:
                        continue
                    
                    ## 이미 지나간 경로인지 확인
                    if (next_x_idx, next_y_idx, next_yaw_idx) in visited:
                        continue
                    visited.add((next_x_idx, next_y_idx, next_yaw_idx))

                    new_h = self.hmap[next_y_idx][next_x_idx]
                    step_cost = abs(distance)
                    
                    if abs(next_steer - node.steer) > 0.001:
                        step_cost += abs(distance)/2 ## 핸들조작 패널티
                    if v < 0:
                        step_cost += abs(distance) ## 기어조작 패널티
                    
                    dist_to_obs = self.collision_map[next_y_idx][next_x_idx]

                    safe_buffer = 2.5  # 안전 거리 설정
                    if dist_to_obs < safe_buffer:
                        penalty = (safe_buffer - dist_to_obs) * 20.0 # 벽과 가까우면 패널티 부여
                        step_cost += penalty

                    new_cost = node.cost + step_cost
                    next_node = Node(next_x, next_y, next_yaw, next_x_idx, next_y_idx, next_yaw_idx, next_steer, next_v, new_cost ,node) # x, y, yaw, x_idx, y_idx, yaw_idx, steer, v, cost, parent
                    
                    heapq.heappush(pq, (new_h+new_cost, new_cost, next_node))
        
    
## =========================================================================================================================================================================


























    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""
        if not self.waypoints:
            self.compute_path(obs)
            if not self.waypoints:
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # OBS 불러오기
        v = float(obs.get("state", {}).get("v", 0.0))
        x = obs["state"]["x"] - self.map_extent[0] ## 0 ~ max_x로 범위 제한
        y = self.map_extent[3] - obs["state"]["y"] ## 0 ~ max_y로 범위 제한
        yaw = -obs["state"]["yaw"] 
        target_x1, target_x2, target_y1, target_y2 = obs["target_slot"]
        target_x, target_y = (target_x1 + target_x2) / 2 - self.map_extent[0], self.map_extent[3] - (target_y1 + target_y2)/2
        dt = obs["limits"]["dt"]
        L = obs["limits"]["L"]
        maxSteer = obs["limits"]["maxSteer"]
        maxAccel = obs["limits"]["maxAccel"]
        maxBrake = obs["limits"]["maxBrake"]
        steerRate = obs["limits"]["steerRate"]

        # 조향각 구하기 Stanley 알고리즘 사용

        front_x = x + L * math.cos(yaw)
        front_y = y + L * math.sin(yaw)

        min_dist = float('inf')
        start_search = self.cur_idx
        end_search = min(len(self.waypoints), self.cur_idx + 10)

        for i in range(start_search, end_search):
            wx, wy = self.waypoints[i]
            dist = math.hypot(front_x - wx, front_y - wy)
            if dist < min_dist:
                min_dist = dist
                self.cur_idx = i

        tx, ty = self.waypoints[self.cur_idx]
        nx, ny = self.waypoints[self.cur_idx+1] if self.cur_idx+1 < len(self.waypoints)-1 else (tx, ty)
        k=10  # Stanley 제어기의 이득 상수

        path_yaw = math.atan2(ny - ty, nx - tx)
        cte = -math.sin(path_yaw) * (front_x - tx) + math.cos(path_yaw) * (front_y - ty)

        angle_to_path = path_yaw - yaw
        angle_to_path = (angle_to_path + math.pi) % (2 * math.pi) - math.pi
        new_steer = angle_to_path - math.atan2(k * cte, max(v, 1e-5))
        new_steer = new_steer if maxSteer > abs(new_steer) else maxSteer if new_steer > 0 else -maxSteer
        
        # 속도 제어 (간단한 P 제어기)
        final_x, final_y = self.waypoints[-1]
        center_offset = L * 0.5  # 앞바퀴 중심 오프셋
        center_x = x + center_offset * math.cos(yaw)
        center_y = y + center_offset * math.sin(yaw)

        dist_to_goal = math.hypot(final_x - center_x, final_y - center_y)

        if dist_to_goal > 5.0:
            target_v = 3.0
        elif dist_to_goal > 2.0:
            target_v = 2.0
        elif dist_to_goal > 0.5:
            target_v = 1.5
        else:
            target_v = 0.0

        current_v = v
        speed_error = target_v - abs(current_v)

        Kp = 1.0  # 튜닝 상수
        cmd = Kp * speed_error

        accel = 0.0
        brake = 0.0

        if cmd > 0:
            accel = cmd
            accel = min(max(accel, 0.0), maxAccel)
            brake = 0.0
        else:
            brake = -cmd
            accel = 0.0
            brake = min(max(brake, 0.0), maxBrake)

        if dist_to_goal < 0.3:
            accel = 0.0
            brake = maxBrake
        
        return {
            "steer": float(-new_steer),
            "accel": float(accel),
            "brake": float(brake),
            "gear": "D"
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

