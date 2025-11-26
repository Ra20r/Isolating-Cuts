from collections import deque

class Dinic:
    # define an Edge class inside Dinic for convenience; not to be used globally
    class Edge:
        __slots__ = ("v", "rev", "cap")

        def __init__(self, v, rev, cap):
            self.v = v
            self.rev = rev
            self.cap = cap

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        # forward edge index = len(g[u]), backward edge index = len(g[v])
        self.g[u].append(Dinic.Edge(v, len(self.g[v]), cap))
        self.g[v].append(Dinic.Edge(u, len(self.g[u]) - 1, 0.0))

    def bfs_level(self, s, t, level):
        for i in range(len(level)):
            level[i] = -1
        q = deque()
        level[s] = 0
        q.append(s)
        while q:
            u = q.popleft()
            for e in self.g[u]:
                if e.cap > 0 and level[e.v] < 0:
                    level[e.v] = level[u] + 1
                    if e.v == t:
                        return True
                    q.append(e.v)
        return level[t] >= 0

    def dfs_block(self, u, t, f, level, it):
        if u == t:
            return f
        for i in range(it[u], len(self.g[u])):
            e = self.g[u][i]
            if e.cap > 0 and level[e.v] == level[u] + 1:
                pushed = self.dfs_block(e.v, t, min(f, e.cap), level, it)
                if pushed > 0:
                    e.cap -= pushed
                    self.g[e.v][e.rev].cap += pushed
                    return pushed
            it[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0.0
        level = [-1] * self.n
        # repeat BFS and DFS blocking flow
        while self.bfs_level(s, t, level):
            it = [0] * self.n
            pushed = self.dfs_block(s, t, float('inf'), level, it)
            while pushed and pushed > 0:
                flow += pushed
                pushed = self.dfs_block(s, t, float('inf'), level, it)
        return flow

    def mincut_source_side(self, s):
        # After running max_flow, find vertices reachable from s in residual graph
        seen = [False] * self.n
        q = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            for e in self.g[u]:
                if e.cap > 1e-12 and not seen[e.v]:
                    seen[e.v] = True
                    q.append(e.v)
        return seen
