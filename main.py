"""fishing_tool.py  依赖: pip install opencv-python mss Pillow numpy pyautogui keyboard"""
import sys, os, json, time, threading, random, tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2, numpy as np, mss, pyautogui, keyboard

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# ---------- 配置 ----------
def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("加载 config.json 失败，使用默认值:", e)
        return {}

cfg = load_config()
params = {
    "cast_hold_time": 3.0, "post_cast_wait": 0.5, "max_bite_wait": 30,
    "pointer_loss_time": 0.3, "min_press_time": 3.0, "min_release_time": 1.0,
    "reel_end_wait": 1.8, "short_press_time": 0.2, "next_cast_sleep": 2.0,
    "bite_diff_threshold": 5, "bite_confirm_frames": 1, "cast_adjust_a_time": 0.25,
    "max_reel_time": 29, "post_fail_cooldown": 4.0, "bite_rearm_delay": 4.0,
    "bite_arm_after_cast_delay": 4.0, "reel_sleep_jitter": 0.02,
}
for k in params:
    if k in cfg:
        params[k] = cfg[k]
pointer_region = cfg.get("pointer_region", {"top": 1261, "left": 772, "width": 957, "height": 346})
bite_region   = cfg.get("bite_region",   {"top": 1435, "left": 2290, "width": 38, "height": 32})
LOWER_ORANGE = np.array([8, 150, 200])
UPPER_ORANGE = np.array([28, 255, 255])

# ---------- 全局状态 ----------
running_detection = False
automation_running = False
manual_bite_flag  = False
last_action_text  = "Idle"
bite_detection_enabled = False
last_reel_success_time = 0.0
last_cast_time = 0.0
sct = mss.mss()

def resource_path(rel):
    return os.path.join(getattr(sys, '_MEIPASS', os.path.abspath(".")), rel)

def jitter_sleep(base_sec):
    delta = random.uniform(-params["reel_sleep_jitter"], params["reel_sleep_jitter"])
    time.sleep(max(0.0, base_sec + delta))

def save_config():
    data = {
        "pointer_region": pointer_region,
        "bite_region": bite_region,
        **{k: params[k] for k in params}
    }
    try:
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print("保存配置失败:", e)
        return False

# ---------- 图像处理 ----------
def rotate_image(img, angle):
    h, w = img.shape[:2]
    c = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)

def detect_pointer_angle_and_annotate(bgr_img, ui_handle):
    img = np.ascontiguousarray(bgr_img, dtype=np.uint8)
    if ui_handle.pointer_template is None:
        return None, img, None
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_img = cv2.inRange(hsv_img, LOWER_ORANGE, UPPER_ORANGE)
    img_masked = cv2.bitwise_and(img, img, mask=mask_img)
    best_val, best_loc, best_angle = -1, None, 0
    for ang in range(-40, 41, 10):
        tpl_rot = rotate_image(ui_handle.pointer_template, ang)
        hsv_tpl = cv2.cvtColor(tpl_rot, cv2.COLOR_BGR2HSV)
        mask_tpl = cv2.inRange(hsv_tpl, LOWER_ORANGE, UPPER_ORANGE)
        tpl_masked = cv2.bitwise_and(tpl_rot, tpl_rot, mask=mask_tpl)
        if tpl_masked.shape[2] != img_masked.shape[2]:
            tpl_masked = cv2.cvtColor(tpl_masked, cv2.COLOR_GRAY2BGR)
        res = cv2.matchTemplate(img_masked, tpl_masked, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val, best_loc, best_angle = max_val, max_loc, ang
    if best_val < 0.6:
        return None, img, mask_img
    h, w = tpl_masked.shape[:2]
    cx, cy = best_loc[0] + w // 2, best_loc[1] + h // 2
    cv2.rectangle(img, best_loc, (best_loc[0] + w, best_loc[1] + h), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 4, (255, 0, 0), -1)
    img_h, img_w = img.shape[:2]
    dx, dy = cx - img_w // 2, img_h // 2 - cy
    angle = np.degrees(np.arctan2(dy, dx))
    return angle, img, mask_img

# ---------- 区域选择 ----------
def select_region_via_drag():
    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))[:, :, :3].copy()
    clone = img.copy()
    win = "拖动选择区域 - 按回车确认，ESC取消"
    rect = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "drawing": False}
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            rect["drawing"], rect["x1"], rect["y1"] = True, x, y
        elif event == cv2.EVENT_MOUSEMOVE and rect["drawing"]:
            rect["x2"], rect["y2"] = x, y
            temp = clone.copy()
            cv2.rectangle(temp, (rect["x1"], rect["y1"]), (x, y), (0, 255, 0), 2)
            cv2.imshow(win, temp)
        elif event == cv2.EVENT_LBUTTONUP:
            rect["drawing"] = False
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse)
    cv2.imshow(win, clone)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 13: break
        if k == 27:
            cv2.destroyWindow(win)
            return None
    cv2.destroyWindow(win)
    x1, y1, x2, y2 = rect["x1"], rect["y1"], rect["x2"], rect["y2"]
    left, top = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    return {"top": top, "left": left, "width": w, "height": h} if w and h else None

# ---------- 咬钩检测 ----------
last_bite_gray = None
bite_change_count = 0
def detect_bite_change(sct_thread):
    global last_bite_gray, bite_change_count
    if not bite_detection_enabled: return False
    t = time.time()
    if t - last_reel_success_time < params["bite_rearm_delay"]: return False
    if t - last_cast_time < params["bite_arm_after_cast_delay"]: return False
    frame = np.array(sct_thread.grab(bite_region))[:, :, :3].copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if last_bite_gray is None:
        last_bite_gray = gray
        return False
    diff = cv2.absdiff(last_bite_gray, gray)
    score = np.mean(diff)
    last_bite_gray = gray
    if score > params["bite_diff_threshold"]:
        bite_change_count += 1
    else:
        bite_change_count = 0
    return bite_change_count >= params["bite_confirm_frames"]

# ---------- 自动钓鱼线程 ----------
def automation_loop(ui_handle):
    global automation_running, last_action_text, last_bite_gray, bite_change_count
    global bite_detection_enabled, last_cast_time, last_reel_success_time, manual_bite_flag
    sct_thread = mss.mss()
    automation_running = True
    last_action_text = "Automation started"
    try:
        while automation_running:
            # 1) 抛竿
            last_action_text = "Casting"
            ui_handle.set_last_action(last_action_text)
            pyautogui.mouseDown()
            if params["cast_adjust_a_time"] > 0:
                pre = params["cast_hold_time"] - params["cast_adjust_a_time"]
                if pre > 0: jitter_sleep(pre)
                pyautogui.keyDown('a')
                jitter_sleep(params["cast_adjust_a_time"])
                pyautogui.keyUp('a')
            else:
                jitter_sleep(params["cast_hold_time"])
            pyautogui.mouseUp()
            jitter_sleep(params["post_cast_wait"])
            last_cast_time = time.time()
            bite_detection_enabled = True

            # 2) 等咬钩
            last_action_text = "Waiting for bite..."
            ui_handle.set_last_action(last_action_text)
            start = time.time()
            while automation_running:
                if detect_bite_change(sct_thread) or manual_bite_flag:
                    manual_bite_flag = False
                    print("咬钩 → 收杆")
                    pyautogui.mouseDown(); time.sleep(0.05); pyautogui.mouseUp()
                    last_bite_gray = None; bite_change_count = 0
                    bite_detection_enabled = False
                    break
                if time.time() - start > params["max_bite_wait"]:
                    last_action_text = "No bite (timeout)"
                    ui_handle.set_last_action(last_action_text)
                    bite_detection_enabled = False
                    jitter_sleep(1)
                    break
                jitter_sleep(0.12)
            else:
                continue

            # 3) 收杆
            last_action_text = "Reeling"
            ui_handle.set_last_action(last_action_text)
            reel_start = time.time(); no_pointer_time = None
            while automation_running:
                if time.time() - reel_start > params["max_reel_time"]:
                    print("收杆超时"); pyautogui.mouseUp()
                    jitter_sleep(params["post_fail_cooldown"]); break

                frame = np.array(sct_thread.grab(pointer_region))[:, :, :3].copy()
                angle, _, _ = detect_pointer_angle_and_annotate(frame, ui_handle)
                if angle is None:
                    no_pointer_time = time.time() if no_pointer_time is None else no_pointer_time
                    if time.time() - no_pointer_time > params["pointer_loss_time"]:
                        print("指针丢失 → 成功")
                        pyautogui.mouseUp()
                        last_reel_success_time = time.time()
                        jitter_sleep(params["reel_end_wait"])
                        pyautogui.mouseDown()
                        jitter_sleep(params["short_press_time"])
                        pyautogui.mouseUp()
                        jitter_sleep(params["next_cast_sleep"])
                        break
                    continue
                else:
                    no_pointer_time = None

                # ---- 阻塞式周期按压 ----
                t = time.time()
                period = params["min_press_time"] + params["min_release_time"]
                phase = t % period
                if phase < params["min_press_time"]:
                    pyautogui.mouseDown()
                    time.sleep(params["min_press_time"] - phase)
                    pyautogui.mouseUp()
                else:
                    pyautogui.mouseUp()
                    time.sleep(period - phase)
    finally:
        automation_running = False
        ui_handle.set_last_action("Automation stopped")

# ---------- Tk UI ----------
class FishingUI:
    def __init__(self, root):
        self.root = root
        root.title("钓鱼助手")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 信息栏
        self.info = tk.Label(root, text="角度: - , 速度: - , 状态: 空闲", font=("Consolas", 12))
        self.info.pack(pady=4)

        # 图像显示
        self.img_label = tk.Label(root, bg="gray")
        self.img_label.pack(pady=4)
        self.bite_label = tk.Label(root, bg="gray")
        self.bite_label.pack(pady=4)

        # 参数区
        pf = tk.Frame(root)
        pf.pack(fill="x", padx=6, pady=4)
        self.entries = {}
        self._add_params(pf)

        # 按钮区
        bf = tk.Frame(root)
        bf.pack(fill="x", padx=6, pady=4)
        self.btn_detect = tk.Button(bf, text="开始检测", command=self.toggle_detect)
        self.btn_detect.pack(side="left", padx=3)
        self.btn_auto = tk.Button(bf, text="启动自动钓鱼 (2键)", command=self.toggle_auto)
        self.btn_auto.pack(side="left", padx=3)

        rf = tk.Frame(root)
        rf.pack(fill="x", padx=6, pady=4)
        tk.Button(rf, text="设置指针区域", command=self.set_ptr_reg).pack(side="left", padx=3)
        tk.Button(rf, text="设置咬钩区域", command=self.set_bite_reg).pack(side="left", padx=3)

        # 状态
        self.action_var = tk.StringVar(value="空闲")
        tk.Label(root, textvariable=self.action_var).pack(pady=4)

        # 热键
        keyboard.add_hotkey('2', self.toggle_auto)
        keyboard.add_hotkey('3', self.manual_bite)

        # 模板
        try:
            self.pointer_template = cv2.imread(resource_path("pointer_template.png"), cv2.IMREAD_COLOR)
            if self.pointer_template is None:
                raise FileNotFoundError("pointer_template.png 未找到")
        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.pointer_template = None

        # 检测循环
        self._detecting = False
        self._after_id = None
        self.prev_angle = None
        self.prev_time = None
        self.current_speed = None

        # ---------- 保存配置按钮（右下角，绿色） ----------
        save_btn = tk.Button(root, text="保存配置", command=self.save_cfg, bg="#4caf50", fg="white", font=("Consolas", 10, "bold"))
        save_btn.pack(side="right", anchor="se", padx=8, pady=8)

    # ---------- 参数输入 ----------
    def _add_params(self, parent):
        groups = {
            "抛竿": ["cast_hold_time", "post_cast_wait", "cast_adjust_a_time"],
            "等上钩": ["max_bite_wait", "bite_diff_threshold", "bite_confirm_frames"],
            "收杆": ["pointer_loss_time", "min_press_time", "min_release_time", "max_reel_time", "post_fail_cooldown", "reel_sleep_jitter"],
            "收杆后": ["reel_end_wait", "short_press_time", "next_cast_sleep"],
            "咬钩冷却": ["bite_rearm_delay", "bite_arm_after_cast_delay"],
        }
        col = 0
        for g_name, keys in groups.items():
            lf = tk.LabelFrame(parent, text=g_name, padx=5, pady=5)
            lf.grid(row=0, column=col, padx=3, sticky="n")
            for row, k in enumerate(keys):
                # 中文标签映射
                label_cn = {
                    "cast_hold_time": "抛竿时长(s)",
                    "post_cast_wait": "抛竿后等待(s)",
                    "cast_adjust_a_time": "按A时长(s)",
                    "max_bite_wait": "上钩等待(s)",
                    "bite_diff_threshold": "差分阈值",
                    "bite_confirm_frames": "确认帧数",
                    "pointer_loss_time": "指针丢失(s)",
                    "min_press_time": "按压时间(s)",
                    "min_release_time": "释放时间(s)",
                    "max_reel_time": "最长收杆(s)",
                    "post_fail_cooldown": "失败冷却(s)",
                    "reel_sleep_jitter": "sleep抖动(s)",
                    "reel_end_wait": "收杆等待(s)",
                    "short_press_time": "短按时间(s)",
                    "next_cast_sleep": "下一轮等待(s)",
                    "bite_rearm_delay": "咬钩重禁(s)",
                    "bite_arm_after_cast_delay": "抛竿屏蔽(s)",
                }.get(k, k)
                tk.Label(lf, text=label_cn).grid(row=row, column=0, sticky="w")
                e = tk.Entry(lf, width=7)
                e.insert(0, str(params[k]))
                e.grid(row=row, column=1, sticky="w")
                self.entries[k] = e
            col += 1

    # ---------- 功能 ----------
    def set_last_action(self, text):
        self.action_var.set(text)

    def save_cfg(self):
        for k, e in self.entries.items():
            try:
                params[k] = float(e.get())
            except Exception:
                pass
        if save_config():
            messagebox.showinfo("提示", "已保存")
        else:
            messagebox.showerror("错误", "保存失败")

    def set_ptr_reg(self):
        global pointer_region
        r = select_region_via_drag()
        if r:
            pointer_region = r
            messagebox.showinfo("成功", "指针区域已更新")

    def set_bite_reg(self):
        global bite_region
        r = select_region_via_drag()
        if r:
            bite_region = r
            messagebox.showinfo("成功", "咬钩区域已更新")

    def manual_bite(self):
        global manual_bite_flag
        manual_bite_flag = True

    def toggle_detect(self):
        if self._detecting:
            self._detecting = False
            if self._after_id:
                self.root.after_cancel(self._after_id)
            self.btn_detect.config(text="开始检测")
        else:
            self._detecting = True
            self.btn_detect.config(text="停止检测")
            self.loop_detect()

    def toggle_auto(self):
        global automation_running
        if automation_running:
            automation_running = False
            self.set_last_action("正在停止自动钓鱼...")
        else:
            automation_running = True
            threading.Thread(target=automation_loop, args=(self,), daemon=True).start()
            self.set_last_action("自动钓鱼已启动（2键停止，3键手动咬钩）")

    def loop_detect(self):
        if not self._detecting:
            return
        frame = np.array(sct.grab(pointer_region))[:, :, :3].copy()
        angle, ann, _ = detect_pointer_angle_and_annotate(frame, self)
        bite_frame = np.array(sct.grab(bite_region))[:, :, :3].copy()

        now = time.time()
        speed = "-"
        if angle is not None and self.prev_angle is not None and self.prev_time is not None:
            dt = now - self.prev_time
            speed = f"{(self.prev_angle - angle) / dt:.2f} 度/秒"
        self.info.config(text=f"角度: {angle:.2f}" if angle else "角度: -" + f" , 速度: {speed} , 状态: {'检测中' if self._detecting else '空闲'}")

        # 显示
        for arr, lbl in ((ann, self.img_label), (bite_frame, self.bite_label)):
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((arr.shape[1], arr.shape[0]))
            imgtk = ImageTk.PhotoImage(image=img)
            lbl.imgtk = imgtk
            lbl.config(image=imgtk)

        self.prev_angle, self.prev_time = angle, now
        self._after_id = self.root.after(50, self.loop_detect)

    def on_close(self):
        global automation_running
        automation_running = False
        keyboard.unhook_all_hotkeys()
        self.root.destroy()

# ---------- main ----------
def main():
    root = tk.Tk()
    FishingUI(root)
    print("按键 2：开始/停止自动钓鱼    按键 3：手动触发咬钩")
    root.mainloop()

if __name__ == "__main__":
    main()
