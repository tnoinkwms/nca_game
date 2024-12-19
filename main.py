import numpy as np
import onnxruntime
import random
import pyxel

X_SIZE = 120
Y_SIZE = 120

SCENE_TITLE = 0
SCENE_PLAY = 1
SCENE_GAMEOVER = 2
SCENE_CLEAR = 3
SCENE_TUTORIAL = 4

bullets = []
enemy_seeds = []
gncas = []
BULLET_SPEED = 3
MAX_LIFE = 300
x = 24
y = 6

def update_entities(entities):
    for entity in entities:
        entity.update()

def cleanup_entities(entities):
    for i in range(len(entities) - 1, -1, -1):
        if not entities[i].is_alive:
            del entities[i]

def draw_entities(entities):
    for entity in entities:
        entity.draw()

def closest_color_index(r, g, b,type):
    min_dist = float('inf')
    best_idx = 0
    if type == "lizard" or type =="title":
        c_list = pyxel.colors.to_list()[:]
    else:
        c_list = pyxel.colors.to_list()[:4]
    for i, color_val in enumerate(c_list):
        pr = (color_val >> 16) & 0xFF
        pg = (color_val >> 8) & 0xFF
        pb = color_val & 0xFF
        dr = pr - r
        dg = pg - g
        db = pb - b
        dist = dr*dr + dg*dg + db*db
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx

def draw_nca(lizard, grayscott):
    for row in range(72):
        for col in range(72):
            # Draw grayscott (using scaled coordinates)
            if row < 60 and col < 60:
                r,g,b = [int(val*255) for val in grayscott[row, col, 0:3]]
                c = closest_color_index(r,g,b, "grayscott")
                if c !=3:
                    pyxel.rect(col*2, row*2,2,2, c)

            # Draw lizard
            r, g, b = [int(val*255) for val in lizard[row, col, 1:4]]
            c = closest_color_index(r,g,b,  "lizard")
            if c !=10:
                pyxel.pset(col+x, row+y, c)

def draw_title(title):
    for row in range(60):
        for col in range(60):
            r,g,b = [int(val*255) for val in title[row, col, 0:3]]
            c = closest_color_index(r,g,b, "title")
            #if c !=7:
            pyxel.rect(col*2, row*2,2,2, c)
        

def load_bgm(msc, filename, snd1, snd2, snd3):
    # Loads a json file for 8bit BGM generator by frenchbread.
    # Each track is stored in snd1, snd2 and snd3 of the sound
    # respectively and registered in msc of the music.
    import json
    with open(filename, "rt") as file:
        bgm = json.loads(file.read())
        pyxel.sounds[snd1].set(*bgm[0])
        pyxel.sounds[snd2].set(*bgm[1])
        pyxel.sounds[snd3].set(*bgm[2])
        pyxel.musics[msc].set([snd1], [snd2], [snd3])

class GNCA():
    def __init__(self,position_x, position_y, model_path, height=72, width=72, n_channels=16, agent_type = "enemy"):
        self.height: int = height
        self.width: int = width
        self.x = position_x
        self.y = position_y
        self.n_channels: int = n_channels
        self.session = onnxruntime.InferenceSession(model_path)
        self.input: np.ndarray
        self.output: np.ndarray
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.agent_type = agent_type
        self.make_seeds()
        #gncas.append(self)
    def to_alpha(self, x):
        return np.clip(x[..., 3:4], 0, 0.9999)

    def to_rgba(self, x):
        rgb, a = x[..., :3], self.to_alpha(x)
        return np.concatenate((np.clip(1.0 - a + rgb, 0, 0.9999), a), axis=3)

    def write_alpha_tolist(self, x):
        alpha = self.to_alpha(x).tolist()
        return alpha

    def make_seeds(self):
        x = np.zeros([1, self.height, self.width, self.n_channels], np.float32)
        if self.agent_type == "enemy":
            x[:, self.height // 2 - 1:self.height // 2 + 1, self.width // 2 - 1:self.width // 2 + 1, 3:] = 1.0
        elif self.agent_type == "env":
            pass
        self.input = x
        return x
    
    def run(self) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: self.input})
        self.output = out[0].astype(np.float32)
        self.input = out[0].astype(np.float32)
        return self.input
    
    def update(self):
        self.run()
    def draw(self):
        return self.to_rgba(self.input)[0]
    
class Player():
    def __init__(self, x,y,level):
        self.level = level
        self.x = x
        self.y = y
        self.life = MAX_LIFE
    def update(self):
        if pyxel.btn(pyxel.KEY_LEFT) or pyxel.btn(pyxel.KEY_A):
            self.x -= 3
        elif pyxel.btn(pyxel.KEY_RIGHT) or pyxel.btn(pyxel.KEY_D):
            self.x += 3
        elif pyxel.btn(pyxel.KEY_UP) or pyxel.btn(pyxel.KEY_W):
            self.y -= 3
        elif pyxel.btn(pyxel.KEY_DOWN) or pyxel.btn(pyxel.KEY_S):
            self.y += 3
        if pyxel.btnp(pyxel.KEY_SPACE):
            if self.level == 1:
                Ballet(self.x+4, self.y)
                pyxel.play(3, 10)
            else:
                Ballet(self.x+4, self.y)
                pyxel.play(3, 10)
        if pyxel.btnp(pyxel.KEY_G):
            Ballet(self.x+2, self.y)
            Ballet(self.x+6, self.y)
            pyxel.play(3, 10)
        
        if self.x >= X_SIZE-15:
            self.x =X_SIZE-15
        if self.y >= Y_SIZE-15:
            self.y = Y_SIZE-15
        if self.x <= 0:
            self.x = 0
        if self.y <= 0:
            self.y = 0
    def draw(self):
        pyxel.blt(self.x, self.y, 0, 0, 24, 15, 15,0) 

class Ballet():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = 8
        self.h = 8
        self.damage_count = 0
        self.damage_threshold = 210
        self.is_alive = True
        bullets.append(self)
    def update(self):
        self.y -= BULLET_SPEED
        if self.y + self.h - 1 < 0:
            self.is_alive = False
    def draw(self):
        pyxel.blt(self.x, self.y,0, 8, 16, self.w, self.h,0)

class Enemy_seed():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.h = 8
        self.w = 8
        self.is_alive = True
        enemy_seeds.append(self)
    def update(self):
        self.x += random.randint(-8,8)
        self.y += random.randint(-8,8)
        if self.y + self.h - 1 < 0 or self.x + self.x < 0 or self.x-self.h>X_SIZE or self.y - self.h > Y_SIZE or random.random() > 0.96:
            self.is_alive = False
    def draw(self):
        pyxel.blt(self.x, self.y,0, 16, 16, self.w, self.h, 0)
        
class App():
    def __init__(self):
        pyxel.init(X_SIZE,Y_SIZE)
        pyxel.load("./resource/my_resource.pyxres")
        self.player = Player(55,100,1)
        self.level = 1
        self.hidden_key = 0
        self.th = 0.9
        self.gnca = GNCA(position_x = x, position_y = y,height=72, width=72, model_path="./resource/lizard.onnx", agent_type="enemy")
        self.gs = GNCA(position_x = 0, position_y = 0,height=60, width=60, model_path="./resource/gray_scott.onnx", agent_type = "env")
        self.title = GNCA(position_x = 0, position_y = 0,height=60, width=60, model_path="./resource/logo.onnx",agent_type = "enemy")
        load_bgm(0, "./resource/music.json", 0, 1, 2)
        self.scene = SCENE_TITLE
        pyxel.run(self.update,self.draw)

    def update(self):
        if self.scene == SCENE_TITLE:
            self.update_title_scene()
        elif self.scene == SCENE_PLAY:
            self.update_play_scene()
        elif self.scene == SCENE_GAMEOVER:
            self.update_gameover_scene()
        elif self.scene == SCENE_CLEAR:
            self.update_clear_scene()
        elif self.scene == SCENE_TUTORIAL:
            self.update_tutorial_scene()
    
    def draw(self):
        pyxel.cls(7)
        if self.scene == SCENE_TITLE:
            self.draw_title_scene()
        elif self.scene == SCENE_PLAY:
            self.draw_play_scene()
        elif self.scene == SCENE_GAMEOVER:
            self.draw_gameover_scene()
        elif self.scene == SCENE_CLEAR:
            self.draw_clear_scene()
        elif self.scene == SCENE_TUTORIAL:
            self.draw_tutorial_scene()
    
    def update_title_scene(self):
        self.title.update()
        if pyxel.btnp(pyxel.KEY_SPACE):
            self.hidden_key = 1
        if pyxel.btnp(pyxel.KEY_RETURN):
            self.scene = SCENE_TUTORIAL
            gncas.clear()
    
    def update_tutorial_scene(self):
        if pyxel.btnp(pyxel.KEY_RETURN):
            self.scene = SCENE_PLAY
            pyxel.playm(0, loop=True)
            pyxel.pal()
        
    def update_play_scene(self):
        for bullet in bullets:
            if (bullet.x >= 24) and (bullet.x <= 96) and (bullet.y >y) and (bullet.y) <= 72+y :
                gnca_region = self.gnca.input[0, bullet.y-4-y:bullet.y+4-y, bullet.x-4-x:bullet.x+4-x, 3]
                gnca_nonzero_count = np.count_nonzero(gnca_region)
                bullet.damage_count += gnca_nonzero_count
                self.gnca.input[0, bullet.y-4-y:bullet.y+4-y, bullet.x-4-x:bullet.x+4-x, 3] = 0
                
                if random.random() > 0.93 and gnca_nonzero_count > 0:
                    Enemy_seed(bullet.x, bullet.y)
                    
            gs_region = self.gs.input[0,bullet.y//2-4:bullet.y//2+4,bullet.x//2-4:bullet.x//2+4,3]
            gs_nonzero_count = np.count_nonzero(gs_region)
            bullet.damage_count += gs_nonzero_count
            self.gs.input[0,bullet.y//2-4:bullet.y//2+4,bullet.x//2-4:bullet.x//2+4,3] = 0

            if bullet.damage_count > bullet.damage_threshold:
                bullet.is_alive = False
        
        for seed in enemy_seeds:
            if random.random() > 0.8:
                pyxel.play(3, 11)
                self.gs.input[0, seed.y//2-2:seed.y//2+2,seed.x//2 -2:seed.x//2+2, 3] = 1
        
        player_region = self.gs.input[0,self.player.y//2:(self.player.y+8)//2,self.player.x//2:(self.player.x+8)//2,3]
        life_count = np.count_nonzero(player_region >= self.th)
        self.player.life -= life_count

        enemy_region = self.gnca.input[0,:,:,3]
        enemy_life =  np.count_nonzero(enemy_region)

        if enemy_life <= 0:
            self.scene = SCENE_CLEAR
            pyxel.stop()
            pyxel.play(3, 12)

        if self.player.life <= 0:
            self.scene = SCENE_GAMEOVER
            pyxel.stop()
            pyxel.play(3,12)
        
        self.player.update()
        self.gnca.update()
        self.gs.update()
        update_entities(bullets)
        update_entities(enemy_seeds)
        cleanup_entities(bullets)
        cleanup_entities(enemy_seeds)
    
    def update_gameover_scene(self):
        update_entities(bullets)
        update_entities(enemy_seeds)
        cleanup_entities(bullets)
        cleanup_entities(enemy_seeds)

        if pyxel.btnp(pyxel.KEY_RETURN):
            self.scene = SCENE_PLAY
            enemy_seeds.clear()
            bullets.clear()
            gncas.clear()
            self.level = 1
            self.th == 0.9
            self.player = Player(55,100, self.level)
            self.gnca = GNCA(position_x = x, position_y = y, height=72, width=72,  model_path="./resource/lizard.onnx", agent_type = "enemy")
            self.gs = GNCA(position_x = 0, position_y = 0,height=60, width=60,  model_path="./resource/gray_scott.onnx", agent_type = "env")
            pyxel.playm(0, loop=True)

    def update_clear_scene(self):
        update_entities(bullets)
        update_entities(enemy_seeds)
        cleanup_entities(bullets)
        cleanup_entities(enemy_seeds)
        if pyxel.btnp(pyxel.KEY_RETURN):
            self.scene = SCENE_PLAY
            enemy_seeds.clear()
            bullets.clear()
            gncas.clear()
            self.level +=1
            self.th = 0.6
            #self.player = Player(60,100, self.level)
            self.player.x = 55
            self.player.y = 100
            if self.level >= 2:
                enemy = "./resource/spider.onnx"
                env = "./resource/spider-web.onnx"
            self.gnca = GNCA(position_x = x, position_y = y,height=72, width=72, model_path=enemy, agent_type = "enemy")
            self.gs = GNCA(position_x = 0, position_y = 0,height=60, width=60, model_path=env, agent_type = "env")
            pyxel.playm(0, loop=True)
    
    def draw_title_scene(self):
        title = self.title.draw()
        draw_title(title)
        pyxel.text(30, 100, "PRESS ENTER KEY", pyxel.frame_count % 16)
        #self.title.input[0, pyxel.mouse_y//2-2:pyxel.mouse_y//2+2,pyxel.mouse_x//2-2:pyxel.mouse_x//2+2, :] = 1
        if self.hidden_key == 1:
            self.title.input[0, pyxel.mouse_y//2-2:pyxel.mouse_y//2+2,pyxel.mouse_x//2-2:pyxel.mouse_x//2+2, :] = 1
        #pyxel.text(40,80,"CREATED BY", 7)
        #pyxel.text(28,90, "TAKAHIDE YOSHIDA",7)
        #pyxel.text(38,100, "HIROKI SATO",7)

    def draw_gameover_scene(self):
        pyxel.cls(0)
        #pyxel.blt(10+pyxel.frame_count%60, 100,0,8*(pyxel.frame_count%2),96,8,8, 0)
        #pyxel.blt(30+pyxel.frame_count%60, 110,0,0,88,10,3,7)
        #pyxel.blt(30+pyxel.frame_count%60, 100,0,8*(pyxel.frame_count%5),80,8,8, 0)
        #pyxel.text(70, 110,"loser", 0)
        pyxel.text(43, 40, "YOU DIED", 7)
        pyxel.text(30, 60, "PRESS ENTER KEY", pyxel.frame_count % 16)
    
    def draw_tutorial_scene(self):
        pyxel.cls(0)
        pyxel.blt(20, 10, 0, 16,89, 15, 15, 0)
        pyxel.text(50, 15, "WASD or ARROW",7)

        pyxel.blt(23, 34, 0, 32,96, 8, 8, 0)
        pyxel.text(50, 35, "SPACE",7)

        pyxel.blt(20, 50, 0, 0,24, 16,16,0)
        pyxel.text(50, 55, "THIS IS YOU",7)

        pyxel.blt(20,75,0,16,104, 14, 19, 0)
        pyxel.text(50, 79, "YOUR ENEMY",7)

        pyxel.blt(23,100,0,32,104, 8, 8, 0)
        pyxel.text(50, 100, "FILTHY SLIME",7)

    def draw_clear_scene(self):
        for i in range(2):
            pyxel.blt(random.randint(0,120),random.randint(0,120),0,16,24,8,8,0)
            pyxel.blt(random.randint(0,120),random.randint(0,120),0,24,32,8,8,0)
            pyxel.blt(random.randint(0,120),random.randint(0,120),0,24,24,8,8,0)
            pyxel.blt(random.randint(0,120),random.randint(0,120),0,16,32,8,8,0)
        pyxel.text(22, 40, "You killed them all", 0)
        pyxel.text(30, 60, "PRESS ENTER KEY", pyxel.frame_count % 16)

    def draw_play_scene(self):
        lizard = self.gnca.draw()
        grayscott = self.gs.draw()
        draw_nca(lizard, grayscott)
        draw_entities(bullets)
        draw_entities(enemy_seeds)
        self.player.draw()
        pyxel.text(3, 3, f"LIFE", 8)
        pyxel.text(90, 3, f"STAGE {self.level}", 0)
        pyxel.rect(22,3, MAX_LIFE/10, 5, 10)
        pyxel.rect(22,3, self.player.life//10, 5, 11)
        
App()