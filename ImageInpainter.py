import numpy as np
import heapq
import math
import os
import cv2

'''
# Usage:
# image: BGR image
# mask: gray image
inpainter = ImageInpainter(image)
inpainter.inpaint(mask, radius)
inpainter.saveImage(outpath)
'''

KNOWN = 0 # in the known image area. Its T and I values are known.
BAND = 1 # the pixel belongs to the narrow band. T value ready to update
INSIDE = 2 # in the region to inpaint.
adjacent_4 = [(0,1), (0,-1), (1,0), (-1,0)]
INF = 1.0e6

class ImageInpainter:
    def __init__(self, image):
        self.i = image
        self.l = image.shape[0]
        self.w = image.shape[1]
        self.narrowBand = list()
        self.t = np.full((self.l, self.w), INF, dtype=float)
        self.f = np.full((self.l, self.w), INSIDE, dtype=int)
        self.output = image
        # self.radius = 5
        
    def solve(self, x1, y1, x2, y2, flags):
        # Closed form solution
        if x1 < 0 or x1 >= self.l or y1 < 0 or y1 >= self.w:
            return INF

        if x2 < 0 or x2 >= self.l or y2 < 0 or y2 >= self.w:
            return INF

        flag1 = flags[x1, y1]
        flag2 = flags[x2, y2]

        if flag1 == KNOWN and flag2 == KNOWN:
            dist1 = self.t[x1, y1]
            dist2 = self.t[x2, y2]
            d = 2.0 - (dist1 - dist2) ** 2
            if d > 0.0:
                r = math.sqrt(d)
                s = (dist1 + dist2 - r) / 2.0
                if s >= dist1 and s >= dist2:
                    return s
                s += r
                if s >= dist1 and s >= dist2:
                    return s
                return INF

        if flag1 == KNOWN:
            dist1 = self.t[x1, y1]
            return 1.0 + dist1

        if flag2 == KNOWN:
            dist2 = self.t[x2, y2]
            return 1.0 + dist2
        return INF
    
    def initGraph(self, mask, radius):
        for x in range(self.l):
            for y in range(self.w):
                if mask[x, y] == 0:
                    f = KNOWN
                    for offsetx, offsety in adjacent_4:
                        x_ = x + offsetx
                        y_ = y + offsety
                        if x_ < 0 or y_ < 0 or x_ >= self.l or y_ >= self.w:
                            continue
                        if mask[x_, y_] != 0:
                            f = BAND
                            self.t[x, y] = 0.0
                            heapq.heappush(self.narrowBand, (0.0, x, y))
                            break
                    self.f[x, y] = f
        self.initOutsideT(radius)
                    
    def initOutsideT(self, radius):
        # preparation phase
        narrowBand = self.narrowBand.copy()
        flags = self.f.copy()
        # Reverse flags
        for i in range(self.l):
            for j in range(self.w):
                if flags[i, j] == INSIDE:
                    flags[i, j] = KNOWN
                elif flags[i, j] == KNOWN:
                    flags[i, j] = INSIDE
                    
        last_t = 0.0
        diameter = radius * 2
        while narrowBand:
            if last_t > diameter:
                break
            _, x, y = heapq.heappop(narrowBand)
            flags[x, y] = KNOWN
            for offsetx, offsety in adjacent_4:
                x_ = x + offsetx
                y_ = y + offsety
                if x_ < 0 or y_ < 0 or x_ >= self.l or y_ >= self.w:
                    continue
                if flags[x_, y_] == INSIDE:
                    last_t = min([
                        self.solve(x_ - 1, y_, x_, y_ - 1, flags),\
                        self.solve(x_ - 1, y_, x_, y_ + 1, flags),\
                        self.solve(x_ + 1, y_, x_, y_ - 1, flags),\
                        self.solve(x_ + 1, y_, x_, y_ + 1, flags)
                    ])
                    self.t[x_, y_] = -last_t
                    flags[x_, y_] = BAND
                    heapq.heappush(narrowBand, (last_t, x_, y_))
        
        
    
    def tGradient(self, x, y):
        # Calculate gradient T (center estimation)
        curLevel = self.t[x, y]

        prev_x = x - 1
        next_x = x + 1
        if prev_x < 0 or next_x >= self.l:
            grad_x = INF
        else:
            f_prev_x = self.f[prev_x, y]
            f_next_x = self.f[next_x, y]

            if f_prev_x != INSIDE and f_next_x != INSIDE:
                grad_x = (self.t[next_x, y] - self.t[prev_x, y]) / 2.
            elif f_prev_x != INSIDE:
                grad_x = curLevel - self.t[prev_x, y]
            elif f_next_x != INSIDE:
                grad_x = self.t[next_x, y] - curLevel
            else:
                grad_x = 0.0

        prev_y = y - 1
        next_y = y + 1
        if prev_y < 0 or next_y >= self.l:
            grad_y = INF
        else:
            f_prev_y = self.f[x, prev_y]
            f_next_y = self.f[x, next_y]

            if f_prev_y != INSIDE and f_next_y != INSIDE:
                grad_y = (self.t[x, next_y] - self.t[x, prev_y]) / 2.
            elif f_prev_y != INSIDE:
                grad_y = curLevel - self.t[x, prev_y]
            elif f_next_y != INSIDE:
                grad_y = self.t[x, next_y] - curLevel
            else:
                grad_y = 0.0

        return grad_x, grad_y
    
    def inpaintPixel(self, x, y, radius):
        # inpaint a pixel
        t = self.t[x, y]
        t_grad_x, t_grad_y = self.tGradient(x, y)
        pixel_sum = np.zeros((3), dtype=float)
        weight_sum = 0.0

        # Pixels within neighborhood radius
        for x_ in range(x - radius, x + radius + 1):
            if x_ < 0 or x_ >= self.l:
                continue
            for y_ in range(y - radius, y + radius + 1):
                if y_ < 0 or y_ >= self.w:
                    continue

                if self.f[x_, y_] == INSIDE:
                    continue
                dir_vector = np.array([x-x_, y-y_], dtype=float)
                dir_norm = np.linalg.norm(dir_vector)
                
                if dir_norm > radius:
                    continue

                # compute weight factors
                dir = abs(dir_vector[0] * t_grad_x + dir_vector[1] * t_grad_y) / dir_norm
                #print(dir, dir_vector[0],t_grad_x, dir_vector[1], t_grad_y )
                if dir == 0.0:
                    dir = 1e-6

                t_ = self.t[x_, y_]
                lev = 1.0 / (1.0 + abs(t_ - t))

                dis = 1.0 / (dir_norm * dir_norm)

                w = abs(dir * dis * lev)

                pixel_sum[0] += w * self.i[x_, y_, 0]
                pixel_sum[1] += w * self.i[x_, y_, 1]
                pixel_sum[2] += w * self.i[x_, y_, 2]

                weight_sum += w

        self.output[x, y] = (pixel_sum / weight_sum).astype(np.uint8)
    
    def inpaint(self, mask, radius=5):
        if self.i.shape[0:2] != mask.shape[0:2]:
            raise Exception("Image size & mask size mismatch.")
        
        self.initGraph(mask, radius)
        while self.narrowBand:
            _, x, y = heapq.heappop(self.narrowBand)
            self.f[x, y] = KNOWN
            
            for offsetx, offsety in adjacent_4:
                x_ = x + offsetx
                y_ = y + offsety
                if x_ < 0 or y_ < 0 or x_ >= self.l or y_ >= self.w:
                    continue

                if self.f[x_, y_] != INSIDE:
                    continue
                newT = min([
                        self.solve(x_ - 1, y_, x_, y_ - 1, self.f),\
                        self.solve(x_ - 1, y_, x_, y_ + 1, self.f),\
                        self.solve(x_ + 1, y_, x_, y_ - 1, self.f),\
                        self.solve(x_ + 1, y_, x_, y_ + 1, self.f)
                    ])
                self.t[x_, y_] = newT
                
                self.inpaintPixel(x_, y_, radius)
                self.f[x_, y_] = BAND
                heapq.heappush(self.narrowBand, (newT, x_, y_))
                
    def saveImage(self, filepath):
        cv2.imwrite(filepath, self.output)

if __name__ == "__main__":
    filename = "1.jpg"
    datadir = 'data'
    maskdir = 'mask'
    outdir = 'out'
    
    datapath = os.path.join(datadir, filename)
    maskpath = os.path.join(maskdir, filename)
    outpath = os.path.join(outdir, filename)
    image = cv2.imread(datapath)
    mask = cv2.imread(maskpath, 0)
    
    inpainter = ImageInpainter(image)
    inpainter.inpaint(mask, 6)
    inpainter.saveImage(outpath)