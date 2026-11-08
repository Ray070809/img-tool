#!/usr/bin/env python3
"""img_tool.py - 读取图片信息并进行简单编辑的命令行工具。

所有编辑结果都保存为新文件，不会覆盖原图。
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 读取与信息
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """读取图片，支持中文路径。返回 BGR / BGRA / 灰度 ndarray。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"无法读取文件：{path}（{exc}）") from exc
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"不是有效的图片文件：{path}")
    return img


def detect_format(path: str) -> str:
    """优先用 Pillow 识别真实格式，失败时回退到扩展名。"""
    try:
        with Image.open(path) as im:
            return im.format or "未知"
    except Exception:
        ext = os.path.splitext(path)[1].lstrip(".").upper()
        return ext or "未知"


def describe_color_mode(img: np.ndarray) -> str:
    """根据通道数给出颜色模式描述。"""
    if img.ndim == 2:
        return "灰度 (Grayscale)"
    channels = img.shape[2]
    return {3: "RGB", 4: "RGBA"}.get(channels, f"{channels} 通道")


def format_file_size(num_bytes: int) -> str:
    """把字节数格式化为 KB / MB，保留 2 位小数。"""
    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.2f} KB"
    return f"{kb / 1024:.2f} MB"


def collect_image_info(path: str, img: np.ndarray) -> dict:
    height, width = img.shape[:2]
    return {
        "文件名": os.path.basename(path),
        "图片格式": detect_format(path),
        "宽度": f"{width} px",
        "高度": f"{height} px",
        "颜色模式": describe_color_mode(img),
        "文件大小": format_file_size(os.path.getsize(path)),
    }


# ---------------------------------------------------------------------------
# 保存与输出路径
# ---------------------------------------------------------------------------

def build_output_path(input_path: str, suffix: str, ext: str | None = None) -> str:
    root, src_ext = os.path.splitext(input_path)
    ext = ext or src_ext
    if not ext.startswith("."):
        ext = "." + ext
    return f"{root}{suffix}{ext}"


def ensure_not_overwrite(input_path: str, output_path: str) -> None:
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("输出路径不能与原图相同，程序不会覆盖原图，请指定其他输出路径")


def save_image(img: np.ndarray, output_path: str) -> None:
    """按扩展名保存图片；JPG 不支持透明通道，先转 RGB。"""
    ext = os.path.splitext(output_path)[1].lower()
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"输出目录不存在：{out_dir}")

    if ext in (".jpg", ".jpeg"):
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == ".png":
        ok, buf = cv2.imencode(".png", img)
    elif ext in (".bmp", ".webp", ".tif", ".tiff"):
        ok, buf = cv2.imencode(ext, img)
    else:
        raise ValueError(f"不支持的输出格式：{ext or '（无扩展名）'}")

    if not ok:
        raise ValueError(f"保存失败：{output_path}")
    buf.tofile(output_path)


# ---------------------------------------------------------------------------
# 编辑功能
# ---------------------------------------------------------------------------

def resize_image(img: np.ndarray, width: int | None, height: int | None) -> np.ndarray:
    h, w = img.shape[:2]
    if width is None and height is None:
        raise ValueError("缩放需要至少指定 --width 或 --height")
    if width is not None and width <= 0:
        raise ValueError("宽度必须大于 0")
    if height is not None and height <= 0:
        raise ValueError("高度必须大于 0")

    if width is None:
        width = max(1, round(w * height / h))
    if height is None:
        height = max(1, round(h * width / w))

    interp = cv2.INTER_AREA if (width < w or height < h) else cv2.INTER_LINEAR
    return cv2.resize(img, (width, height), interpolation=interp)


def crop_image(img: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    h, w = img.shape[:2]
    if left < 0 or top < 0:
        raise ValueError("裁剪坐标不能为负数")
    if right <= left or bottom <= top:
        raise ValueError("裁剪区域无效：右边界必须大于左边界，下边界必须大于上边界")
    if right > w or bottom > h:
        raise ValueError(
            f"裁剪区域越界：图片尺寸为 {w}x{h}，右边界应 ≤ {w}，下边界应 ≤ {h}"
        )
    return img[top:bottom, left:right]


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    angle = float(angle) % 360
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 任意角度：扩展画布，避免边角被裁掉
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    if img.ndim == 2:
        border = 255
    elif img.shape[2] == 4:
        border = (255, 255, 255, 0)
    else:
        border = (255, 255, 255)
    return cv2.warpAffine(img, matrix, (new_w, new_h), borderValue=border)


def flip_image(img: np.ndarray, direction: str) -> np.ndarray:
    if direction == "horizontal":
        return cv2.flip(img, 1)
    if direction == "vertical":
        return cv2.flip(img, 0)
    raise ValueError("翻转方向只能是 horizontal 或 vertical")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    img = load_image(args.image)
    for key, value in collect_image_info(args.image, img).items():
        print(f"{key}: {value}")
    return 0


def cmd_resize(args: argparse.Namespace) -> int:
    img = load_image(args.image)
    result = resize_image(img, args.width, args.height)
    out = args.output or build_output_path(args.image, "_resize")
    ensure_not_overwrite(args.image, out)
    save_image(result, out)
    print(f"已保存到：{os.path.abspath(out)}")
    return 0


def cmd_crop(args: argparse.Namespace) -> int:
    img = load_image(args.image)
    result = crop_image(img, args.left, args.top, args.right, args.bottom)
    out = args.output or build_output_path(args.image, "_crop")
    ensure_not_overwrite(args.image, out)
    save_image(result, out)
    print(f"已保存到：{os.path.abspath(out)}")
    return 0


def cmd_rotate(args: argparse.Namespace) -> int:
    img = load_image(args.image)
    result = rotate_image(img, args.angle)
    out = args.output or build_output_path(args.image, "_rotated")
    ensure_not_overwrite(args.image, out)
    save_image(result, out)
    print(f"已保存到：{os.path.abspath(out)}")
    return 0


def cmd_flip(args: argparse.Namespace) -> int:
    img = load_image(args.image)
    result = flip_image(img, args.direction)
    out = args.output or build_output_path(args.image, "_flip")
    ensure_not_overwrite(args.image, out)
    save_image(result, out)
    print(f"已保存到：{os.path.abspath(out)}")
    return 0


def cmd_grayscale(args: argparse.Namespace) -> int:
    img = load_image(args.image)
    result = to_grayscale(img)
    out = args.output or build_output_path(args.image, "_gray")
    ensure_not_overwrite(args.image, out)
    save_image(result, out)
    print(f"已保存到：{os.path.abspath(out)}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    img = load_image(args.image)
    ext = "." + args.format.lower()
    out = args.output or build_output_path(args.image, "_converted", ext)
    ensure_not_overwrite(args.image, out)
    save_image(img, out)
    print(f"已保存到：{os.path.abspath(out)}")
    return 0


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="img_tool.py",
        description="读取图片信息并进行简单编辑（不覆盖原图）。",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="命令")

    p = sub.add_parser("info", help="读取图片基础信息")
    p.add_argument("image", help="图片路径")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("resize", help="缩放图片")
    p.add_argument("image", help="图片路径")
    p.add_argument("--width", type=int, help="目标宽度（像素）")
    p.add_argument("--height", type=int, help="目标高度（像素）")
    p.add_argument("-o", "--output", help="输出文件路径")
    p.set_defaults(func=cmd_resize)

    p = sub.add_parser("crop", help="裁剪图片")
    p.add_argument("image", help="图片路径")
    p.add_argument("--left", type=int, required=True, help="左边界 x")
    p.add_argument("--top", type=int, required=True, help="上边界 y")
    p.add_argument("--right", type=int, required=True, help="右边界 x")
    p.add_argument("--bottom", type=int, required=True, help="下边界 y")
    p.add_argument("-o", "--output", help="输出文件路径")
    p.set_defaults(func=cmd_crop)

    p = sub.add_parser("rotate", help="旋转图片")
    p.add_argument("image", help="图片路径")
    p.add_argument("--angle", type=float, required=True, help="旋转角度，如 90、180、270")
    p.add_argument("-o", "--output", help="输出文件路径")
    p.set_defaults(func=cmd_rotate)

    p = sub.add_parser("flip", help="翻转图片")
    p.add_argument("image", help="图片路径")
    p.add_argument("--direction", choices=["horizontal", "vertical"], required=True,
                   help="horizontal=水平翻转，vertical=垂直翻转")
    p.add_argument("-o", "--output", help="输出文件路径")
    p.set_defaults(func=cmd_flip)

    p = sub.add_parser("grayscale", help="灰度化")
    p.add_argument("image", help="图片路径")
    p.add_argument("-o", "--output", help="输出文件路径")
    p.set_defaults(func=cmd_grayscale)

    p = sub.add_parser("convert", help="格式转换")
    p.add_argument("image", help="图片路径")
    p.add_argument("--format", choices=["jpg", "jpeg", "png"], required=True,
                   help="目标格式")
    p.add_argument("-o", "--output", help="输出文件路径")
    p.set_defaults(func=cmd_convert)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, OSError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())