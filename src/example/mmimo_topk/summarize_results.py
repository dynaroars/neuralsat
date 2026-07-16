"""
manifest.csv(idx,eps,net,data,k,abs_row,result)와 run_batch.py가 만든 result
파일들을 모아서 summary.csv(idx,eps,status,runtime)와 idx별 certified-robust
반경을 robustness_summary.csv로 정리하고, eps별 unsat/sat 비율을 floating
range 차트(I-beam 스타일)로 eps_unsat_ratio.png(Pillow만 사용, 새 의존성 없음)에
저장한다.

robust radius 정의: eps를 오름차순으로 볼 때, 가장 작은 eps부터 연속으로
unsat인 구간의 마지막 eps (= 그보다 작은 eps는 전부 unsat으로 확인된 반경).
중간에 unsat이 아닌 결과가 나온 뒤 더 큰 eps에서 다시 unsat이 나오면(비단조),
anomaly로 표시한다 (L_inf eps-ball이 커질수록 sat/unknown 쪽으로 가는게
자연스러운데 그 가정이 깨진 경우).
"""

from __future__ import annotations

import argparse
import csv
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "vnnlib" / "manifest.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "summary.csv"
DEFAULT_ROBUSTNESS_SUMMARY = SCRIPT_DIR / "robustness_summary.csv"
DEFAULT_CHART = SCRIPT_DIR / "eps_unsat_ratio.png"

_CHART_INK = (11, 11, 11)
_CHART_MUTED = (137, 135, 129)
_CHART_BASELINE = (195, 194, 183)
_LINE_UNSAT = (42, 120, 214)
_LINE_SAT = (27, 175, 122)
_CHART_BG = (252, 252, 251)


def _chart_font(size: int):
    from PIL import ImageFont

    # 한글 라벨이 있어서 malgun.ttf(맑은 고딕, Windows 기본 탑재)를 우선 시도하고,
    # 없는 환경(리눅스 등)이면 arial -> Pillow 내장 비트맵 폰트 순으로 대체한다.
    for name in ("malgun.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _i_beam(draw, cx, y0, y1, color, cap_w=34, width=3):
    """세로선 + 위아래 가로 캡(I자 모양)의 floating range 마크 (범례용)."""
    draw.line([(cx, y0), (cx, y1)], fill=color, width=width)
    draw.line([(cx - cap_w / 2, y0), (cx + cap_w / 2, y0)], fill=color, width=width)
    draw.line([(cx - cap_w / 2, y1), (cx + cap_w / 2, y1)], fill=color, width=width)


def _split_beam(draw, cx, bottom_y, boundary_y, top_y, color_bottom, color_top, cap_w=34, width=3):
    """세로선 하나가 경계에서 색만 바뀌는 형태. 캡은 맨 위/맨 아래에만 그려서
    간격이 있다 없다 하는 지저분함 없이 항상 이어 붙게 만든다."""
    draw.line([(cx, bottom_y), (cx, boundary_y)], fill=color_bottom, width=width)
    draw.line([(cx, boundary_y), (cx, top_y)], fill=color_top, width=width)
    draw.line([(cx - cap_w / 2, bottom_y), (cx + cap_w / 2, bottom_y)], fill=color_bottom, width=width)
    draw.line([(cx - cap_w / 2, top_y), (cx + cap_w / 2, top_y)], fill=color_top, width=width)


def write_eps_ratio_chart(records: list[dict], out_path: pathlib.Path) -> None:
    """eps별 unsat/sat 비율(전체 idx 대비)을 floating range 차트(I-beam 스타일)로
    PNG 이미지로 저장한다 (Pillow만 사용, 새 의존성 없음)."""
    from PIL import Image, ImageDraw

    by_eps: dict[float, dict[str, int]] = {}
    for r in records:
        d = by_eps.setdefault(r["eps"], {"unsat": 0, "total": 0})
        d["total"] += 1
        if r["status"] == "unsat":
            d["unsat"] += 1

    eps_sorted = sorted(by_eps)
    n_idx = max((d["total"] for d in by_eps.values()), default=0)
    n = len(eps_sorted)

    margin_l, margin_r, margin_t, margin_b = 60, 70, 90, 60
    width, height = 720, 480
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    baseline_y = margin_t + chart_h
    top_y = margin_t

    img = Image.new("RGB", (width, height), _CHART_BG)
    draw = ImageDraw.Draw(img)

    font_title = _chart_font(20)
    font_subtitle = _chart_font(13)
    font_label = _chart_font(12)
    font_small = _chart_font(12)

    draw.text((margin_l, 16), f"eps별 unsat / sat 비율 (전체 {n_idx}개 idx 대비)", fill=_CHART_INK, font=font_title)
    draw.text((margin_l, 42), "아래(파랑)=unsat 구간, 위(초록)=sat 구간 (둘을 합치면 항상 0~100%)", fill=_CHART_MUTED, font=font_subtitle)

    # legend
    lx, ly = margin_l, 64
    _i_beam(draw, lx + 10, ly, ly + 14, _LINE_UNSAT, cap_w=16, width=2)
    draw.text((lx + 26, ly + 1), "unsat", fill=_CHART_MUTED, font=font_small)
    lx2 = lx + 26 + draw.textlength("unsat", font=font_small) + 26
    _i_beam(draw, lx2 + 10, ly, ly + 14, _LINE_SAT, cap_w=16, width=2)
    draw.text((lx2 + 26, ly + 1), "sat", fill=_CHART_MUTED, font=font_small)

    xs = [margin_l + i * (chart_w / (n - 1)) for i in range(n)] if n > 1 else [margin_l + chart_w / 2]

    for x, eps in zip(xs, eps_sorted):
        d = by_eps[eps]
        unsat_ratio = d["unsat"] / d["total"] * 100 if d["total"] else 0.0
        sat_ratio = 100.0 - unsat_ratio
        boundary_y = baseline_y - (unsat_ratio / 100) * chart_h

        _split_beam(draw, x, baseline_y, boundary_y, top_y, _LINE_UNSAT, _LINE_SAT)

        u_label = f"{unsat_ratio:.1f}%"
        s_label = f"{sat_ratio:.1f}%"
        tw_u = draw.textlength(u_label, font=font_label)
        text_h = font_label.size + 2

        # 라벨은 경계(boundary) 높이에 두되, 선/캡과 겹치지 않도록 좌우로 살짝 띄운다.
        # (unsat=파랑은 왼쪽, sat=초록은 오른쪽). 캔버스 밖으로 나가지 않도록 clamp.
        label_cy = max(top_y + text_h / 2 + 4, min(baseline_y - text_h / 2 - 4, boundary_y))
        label_y = label_cy - text_h / 2
        side_gap = 6
        draw.text((x - side_gap - tw_u, label_y), u_label, fill=_LINE_UNSAT, font=font_label)
        draw.text((x + side_gap, label_y), s_label, fill=_LINE_SAT, font=font_label)

        eps_label = f"{eps:g}"
        tw2 = draw.textlength(eps_label, font=font_small)
        draw.text((x - tw2 / 2, baseline_y + 10), eps_label, fill=_CHART_MUTED, font=font_small)

    draw.line([(margin_l, baseline_y), (margin_l + chart_w, baseline_y)], fill=_CHART_BASELINE, width=1)

    draw.text((margin_l, height - 22), "x축: eps (L_inf, 균등 간격 배치)  ·  y축: 비율(%)", fill=_CHART_MUTED, font=font_small)

    img.save(out_path)


def read_result(result_path: pathlib.Path) -> tuple[str, str]:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return "not_run", ""
    first_line = result_path.read_text().splitlines()[0]
    parts = first_line.split(",", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--robustness-out", default=str(DEFAULT_ROBUSTNESS_SUMMARY))
    parser.add_argument("--chart-out", default=str(DEFAULT_CHART), help="eps별 unsat/sat 비율 range 차트(PNG) 출력 경로")
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        status, runtime = read_result(pathlib.Path(row["result"]))
        records.append({"idx": int(row["idx"]), "eps": float(row["eps"]), "status": status, "runtime": runtime})

    with open(args.summary_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "eps", "status", "runtime"])
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r["idx"], r["eps"])):
            writer.writerow(r)

    by_idx: dict[int, list[dict]] = {}
    for r in records:
        by_idx.setdefault(r["idx"], []).append(r)

    robustness_rows = []
    status_counts: dict[str, int] = {}
    for idx, recs in sorted(by_idx.items()):
        recs.sort(key=lambda r: r["eps"])
        robust_radius = None
        break_eps = None
        break_status = None
        anomaly = False
        broke = False
        for r in recs:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            if not broke:
                if r["status"] == "unsat":
                    robust_radius = r["eps"]
                else:
                    broke = True
                    break_eps = r["eps"]
                    break_status = r["status"]
            elif r["status"] == "unsat":
                anomaly = True

        robustness_rows.append(
            {
                "idx": idx,
                "robust_radius_eps": robust_radius if robust_radius is not None else "",
                "first_non_unsat_eps": break_eps if break_eps is not None else "",
                "first_non_unsat_status": break_status if break_status is not None else "",
                "anomaly_nonmonotonic": anomaly,
            }
        )

    with open(args.robustness_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "robust_radius_eps", "first_non_unsat_eps", "first_non_unsat_status", "anomaly_nonmonotonic"],
        )
        writer.writeheader()
        writer.writerows(robustness_rows)

    write_eps_ratio_chart(records, pathlib.Path(args.chart_out))

    print(f"summary: {len(records)} instances across {len(by_idx)} data points")
    print(f"status counts: {status_counts}")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.robustness_out}")
    print(f"wrote {args.chart_out}")


if __name__ == "__main__":
    main()
