# 分辨率检查器已禁用（v1.2.0）：
# 它在每个任务 Starting 时强制截图，冷启动阶段游戏尚未拉起、虚拟屏为空，
# 截图必然失败或返回异常尺寸，进而触发 tasker.post_stop() 把整个任务停掉，
# 这正是多开冷启动时任务秒级"全部完成"的元凶。本项目分辨率固定 1280x720，
# 无需运行时检查。文件保留以便日后回滚，只要不 import，装饰器就不会注册。
# from .aspect_ratio import AspectRatioChecker  # noqa: F401

# 取消渲染模式检查器（本项目跑真机，不是 MuMu 模拟器，该检查本就无意义）
# from .render_mode_checker import MuMuRenderChecker