#!/usr/bin/env python
"""
v3 적용 전 sanity check.

세션 객체에 depression_severity, anxiety_severity, addiction_severity가
정확히 그 이름으로 존재하는지 확인. 이름이 다르면 features.py에서 0이 들어가
v3 효과가 없어지므로 사전 검증 필수.

실행:
    python sanity_check_v3.py --extracted data/extracted
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, 'src')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted', required=True, type=str)
    parser.add_argument('--split', default='training')
    args = parser.parse_args()

    from data.parser import parse_all_sessions

    print("[1] 첫 세션 파싱")
    labeling = Path(args.extracted) / args.split / 'labeling'
    sessions = parse_all_sessions(labeling, progress=False)
    if not sessions:
        sys.exit("ERROR: 세션 없음")
    print(f"  → {len(sessions):,} 세션 로드")

    # 첫 5개 세션의 severity attribute 확인
    print("\n[2] Severity attribute 검증 (첫 5개 세션)")
    print(f"  {'patient':<10s} {'session':<8s} {'dep':<6s} {'anx':<6s} {'add':<6s}")
    print(f"  {'-'*40}")
    n_ok = 0
    for s in sessions[:5]:
        try:
            dep = float(getattr(s, 'depression_severity', None))
            anx = float(getattr(s, 'anxiety_severity', None))
            add = float(getattr(s, 'addiction_severity', None))
            print(f"  {s.patient_id:<10s} s{s.session_number:<6d} {dep:<6.1f} {anx:<6.1f} {add:<6.1f}")
            n_ok += 1
        except (TypeError, ValueError) as e:
            print(f"  {s.patient_id} s{s.session_number}: ERROR {e}")
            # 대안 attribute 이름들 확인
            print(f"    available attributes: ", [a for a in dir(s) if 'sever' in a.lower() or 'depres' in a.lower() or 'anxie' in a.lower() or 'addic' in a.lower()])

    if n_ok == 0:
        print("\n❌ Severity attribute가 없거나 다른 이름임. features.py 수정 필요.")
        # 대안 이름 확인
        sample = sessions[0]
        print(f"\nSession 객체의 모든 attribute (참고용):")
        for attr in sorted(dir(sample)):
            if not attr.startswith('_'):
                try:
                    val = getattr(sample, attr)
                    if not callable(val) and isinstance(val, (int, float, str)):
                        print(f"    {attr} = {val}")
                except Exception:
                    pass
        sys.exit(1)
    else:
        print(f"\n  ✓ {n_ok}/5 세션에서 severity attribute 정상 추출")

    # 전체 세션에서 severity 분포
    print("\n[3] 전체 세션의 severity 분포")
    import numpy as np
    devs, anxs, adds = [], [], []
    for s in sessions:
        try:
            devs.append(float(s.depression_severity))
            anxs.append(float(s.anxiety_severity))
            adds.append(float(s.addiction_severity))
        except Exception:
            pass

    if devs:
        d, a, ad = np.array(devs), np.array(anxs), np.array(adds)
        print(f"  depression : mean={d.mean():.2f}, std={d.std():.2f}, min={d.min():.0f}, max={d.max():.0f}")
        print(f"  anxiety    : mean={a.mean():.2f}, std={a.std():.2f}, min={a.min():.0f}, max={a.max():.0f}")
        print(f"  addiction  : mean={ad.mean():.2f}, std={ad.std():.2f}, min={ad.min():.0f}, max={ad.max():.0f}")

        # 0-3 범위가 맞는지
        max_v = max(d.max(), a.max(), ad.max())
        print(f"\n  최대값: {max_v:.0f}")
        if max_v > 5:
            print(f"  ⚠️  최대값이 5 초과! (예상 0-3) → features.py의 / 3.0 정규화가 맞는지 확인 필요")
        elif max_v <= 3:
            print(f"  ✓ 0-3 범위, 정규화 / 3.0 적합")
        else:
            print(f"  ⚠️  3-5 범위 (sum score일 가능성) → 정규화 / 5.0이 더 맞을 수 있음")

    print("\n✓ Sanity check 완료. v3 패치 적용 안전.\n")


if __name__ == '__main__':
    main()
