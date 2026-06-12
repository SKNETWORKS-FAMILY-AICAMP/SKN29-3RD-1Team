import os, sys, re, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.rag.splitter import split_md_smart, detect_format
from app.rag.vector_store import get_vector_store


def run(folder_path):
    # 1. md 파일 재귀 수집
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in sorted(files):
            if not f.endswith(".md") or "(1)" in f:
                continue
            fpath = os.path.join(root, f)
            fname = f
            if "#U" in f:
                try:
                    fname = f.replace("#U", "\\u").encode().decode('unicode_escape')
                except:
                    fname = f
            all_files.append({"path": fpath, "name": fname, "dir": root})

    if not all_files:
        print(f"md 파일 없음: {folder_path}")
        return

    # 2. blog 파일 제외 (전체 읽어서 source_type 확인)
    targets = []
    blog_count = 0
    for t in all_files:
        with open(t["path"], encoding="utf-8") as f:
            content = f.read()
        source_m = re.search(r'"source_type"\s*:\s*"([^"]+)"', content)
        if source_m and source_m.group(1) == "blog":
            blog_count += 1
            continue
        targets.append(t)

    # 3. complete 파일이 있는 폴더에서 원문 제외
    complete_dirs = set()
    for t in targets:
        if "complete" in t["name"].lower():
            complete_dirs.add(t["dir"])

    final = []
    for t in targets:
        if t["dir"] in complete_dirs and "complete" not in t["name"].lower():
            continue
        final.append(t)

    # 4. 레벨 감지
    def get_level(filepath):
        rel = os.path.relpath(filepath, folder_path)
        for part in rel.replace("\\", "/").split("/"):
            try:
                decoded = part.replace("#U", "\\u").encode().decode('unicode_escape') if "#U" in part else part
            except:
                decoded = part
            if decoded in ["기초", "초급", "중급", "고급"]:
                return decoded
        return "기타"

    print(f"폴더: {folder_path}")
    print(f"전체 md: {len(all_files)}개 | blog 제외: {blog_count}개 | 청킹 대상: {len(final)}개\n")

    level_counts = {}
    for t in final:
        lv = get_level(t["path"])
        level_counts[lv] = level_counts.get(lv, 0) + 1
    for lv in ["기초", "초급", "중급", "고급", "기타"]:
        if lv in level_counts:
            print(f"  [{lv}] {level_counts[lv]}개")
    print()

    # 5. 청킹 + VectorDB 저장
    vector_store = get_vector_store()
    total_chunks = 0

    for t in final:
        filepath = t["path"]
        fname = t["name"]

        try:
            with open(filepath, encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"  [SKIP] {fname} - {e}")
            continue

        # splitter.py가 메타데이터 추출 + 본문 분리 + 청킹 전부 처리
        metadata = {
            "source": os.path.relpath(filepath, folder_path),
            "filename": fname,
            "level": get_level(filepath),
        }

        fmt = detect_format(text)

        # 첫 번째 파일만 3전략 비교 출력
        show = (t == final[0])
        if show:
            print(f"\n  [3전략 비교: {fname}]")
        chunks = split_md_smart(text, metadata, show_comparison=show)

        if chunks:
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  [{fmt}] {fname[:55]:<57} -> {len(chunks):>2}개")

    print(f"\n{'='*60}")
    print(f"완료: {len(final)}개 파일 -> {total_chunks}개 청크 -> chroma_db/")
    print(f"  (blog {blog_count}개 제외)")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('사용법: python run_ingest.py "C:\\skn29\\3차 프로젝트\\자료"')
        sys.exit(1)
    run(sys.argv[1])