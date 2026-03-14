from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "google/embeddinggemma-300m"
DEFAULT_ENTRIES = [
    {"id": 1, "kind": "normal", "op": "+", "text": "私は犬が好きです。"},
    {"id": 2, "kind": "normal", "op": "+", "text": "私は猫も好きです。"},
    {"id": 3, "kind": "composed", "block_id": 1, "normalize": False, "op": "+", "text": "私"},
    {"id": 4, "kind": "composed", "block_id": 1, "normalize": False, "op": "+", "text": "犬"},
    {"id": 5, "kind": "composed", "block_id": 1, "normalize": False, "op": "+", "text": "好き"},
]


st.set_page_config(page_title="Embedding Heatmap Viewer", layout="wide")


def init_state() -> None:
    if "entries" not in st.session_state:
        st.session_state.entries = [item.copy() for item in DEFAULT_ENTRIES]
    next_missing_id = 1
    existing_ids = {entry["id"] for entry in st.session_state.entries if "id" in entry}
    for entry in st.session_state.entries:
        if "id" in entry:
            continue
        while next_missing_id in existing_ids:
            next_missing_id += 1
        entry["id"] = next_missing_id
        existing_ids.add(next_missing_id)
        next_missing_id += 1
    if "new_entry_kind" not in st.session_state:
        st.session_state.new_entry_kind = "normal"
    if "next_entry_id" not in st.session_state:
        existing_entry_ids = [entry.get("id", 0) for entry in st.session_state.entries]
        st.session_state.next_entry_id = (max(existing_entry_ids) if existing_entry_ids else 0) + 1
    if "next_block_id" not in st.session_state:
        existing_block_ids = [
            entry.get("block_id", 0) for entry in st.session_state.entries if entry["kind"] == "composed"
        ]
        st.session_state.next_block_id = (max(existing_block_ids) if existing_block_ids else 0) + 1


def create_entry(kind: str, text: str = "", block_id: int | None = None, normalize: bool = False, op: str = "+") -> dict:
    entry = {
        "id": st.session_state.next_entry_id,
        "kind": kind,
        "op": op,
        "text": text,
    }
    st.session_state.next_entry_id += 1
    if kind == "composed":
        entry["block_id"] = block_id
        entry["normalize"] = normalize
    return entry


@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


def add_entry() -> None:
    kind = st.session_state.new_entry_kind
    if kind == "normal":
        st.session_state.entries.append(create_entry("normal"))
        return

    block_id = st.session_state.next_block_id
    st.session_state.next_block_id += 1
    st.session_state.entries.append(create_entry("composed", block_id=block_id, normalize=False))


def add_entry_to_block(block_id: int) -> None:
    insert_at = len(st.session_state.entries)
    for idx, entry in enumerate(st.session_state.entries):
        if entry["kind"] == "composed" and entry.get("block_id") == block_id:
            insert_at = idx + 1
    st.session_state.entries.insert(insert_at, create_entry("composed", block_id=block_id, normalize=False))


def remove_entry(index: int) -> None:
    if len(st.session_state.entries) > 1:
        st.session_state.entries.pop(index)


def get_groups() -> list[dict[str, int | str]]:
    groups: list[dict[str, int | str]] = []
    idx = 0
    while idx < len(st.session_state.entries):
        entry = st.session_state.entries[idx]
        if entry["kind"] == "normal":
            groups.append({"kind": "normal", "start": idx, "end": idx + 1})
            idx += 1
            continue

        block_id = entry["block_id"]
        start = idx
        while (
            idx < len(st.session_state.entries)
            and st.session_state.entries[idx]["kind"] == "composed"
            and st.session_state.entries[idx].get("block_id") == block_id
        ):
            idx += 1
        groups.append({"kind": "composed", "block_id": block_id, "start": start, "end": idx})
    return groups


def move_group(block_start: int, direction: str) -> None:
    groups = get_groups()
    group_index = next((i for i, group in enumerate(groups) if group["start"] == block_start), None)
    if group_index is None:
        return

    target_index = group_index - 1 if direction == "up" else group_index + 1
    if target_index < 0 or target_index >= len(groups):
        return

    current = groups[group_index]
    target = groups[target_index]
    entries = st.session_state.entries

    current_slice = entries[current["start"] : current["end"]]
    target_slice = entries[target["start"] : target["end"]]

    if direction == "up":
        entries[target["start"] : current["end"]] = current_slice + target_slice
    else:
        entries[current["start"] : target["end"]] = target_slice + current_slice


def embed_texts(model: SentenceTransformer, texts: Iterable[str]) -> np.ndarray:
    return model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )


def shorten_label(text: str, max_len: int = 26) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_len:
        return one_line
    return f"{one_line[: max_len - 1]}…"


def compose_expression(entries: list[dict[str, str]], max_len: int | None = None) -> str:
    parts = []
    for entry in entries:
        text = entry["text"]
        if max_len is not None:
            text = shorten_label(text, max_len=max_len)
        prefix = entry["op"]
        parts.append(f"{prefix}「{text}」")
    return "".join(parts)


def render_heatmap_block(vectors: list[np.ndarray], labels: list[str]) -> None:
    if not vectors:
        return

    matrix = np.stack(vectors, axis=0)
    max_abs = float(np.max(np.abs(matrix))) or 1.0
    fig_height = max(1.1 + len(vectors) * 0.62, 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="bwr",
        center=0.0,
        vmin=-max_abs,
        vmax=max_abs,
        xticklabels=False,
        yticklabels=labels,
        cbar=False,
        linewidths=0.0,
    )

    ax.tick_params(axis="y", labelsize=10, length=0, pad=8)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def compute_cosine_similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = matrix / norms
    return normalized @ normalized.T


def render_cosine_similarity(vectors: list[np.ndarray], labels: list[str]) -> None:
    if len(vectors) < 2:
        return

    similarity = compute_cosine_similarity_matrix(vectors)
    mask = np.tril(np.ones_like(similarity, dtype=bool), k=-1)

    fig_height = max(2.6 + len(labels) * 0.5, 4.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.heatmap(
        similarity,
        mask=mask,
        ax=ax,
        cmap="bwr",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        square=False,
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        cbar_kws={"shrink": 0.8, "label": "Cosine Similarity"},
    )
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    fig.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    pairs: list[tuple[float, str, str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((float(similarity[i, j]), labels[i], labels[j]))
    pairs.sort(key=lambda item: item[0], reverse=True)

    st.caption("類似度が高い組み合わせ")
    top_n = min(3, len(pairs))
    for rank, (score, left, right) in enumerate(pairs[:top_n], start=1):
        st.write(f"{rank}. {left} / {right}: {score:.3f}")


def render_entry_editor() -> None:
    st.subheader("文章入力")
    st.write("通常文章と、足し引き用の文章を同じリストで管理できます。")

    def render_entry_row(idx: int, show_move_buttons: bool = True, move_anchor: int | None = None) -> None:
        entry = st.session_state.entries[idx]
        entry_id = entry["id"]
        anchor = idx if move_anchor is None else move_anchor
        row = st.columns([1.4, 1.1, 8.1, 0.9, 0.9, 1.2])
        with row[0]:
            kind_options = {"normal": "通常", "composed": "足し引き"}
            selected = st.selectbox(
                f"種類 {idx + 1}",
                options=list(kind_options.keys()),
                index=0 if entry["kind"] == "normal" else 1,
                format_func=lambda value: kind_options[value],
                key=f"kind_{entry_id}",
                label_visibility="collapsed",
            )
            if selected == "composed" and "block_id" not in st.session_state.entries[idx]:
                st.session_state.entries[idx]["block_id"] = st.session_state.next_block_id
                st.session_state.entries[idx]["normalize"] = False
                st.session_state.next_block_id += 1
            st.session_state.entries[idx]["kind"] = selected

        with row[1]:
            if st.session_state.entries[idx]["kind"] == "composed":
                current_op = st.session_state.entries[idx]["op"]
                op = st.selectbox(
                    f"演算 {idx + 1}",
                    options=["+", "-"],
                    index=0 if current_op == "+" else 1,
                    key=f"op_{entry_id}",
                    label_visibility="collapsed",
                )
                st.session_state.entries[idx]["op"] = op
            else:
                st.markdown(
                    "<div style='height:38px; display:flex; align-items:center; justify-content:center; color:#6b7280;'>比較</div>",
                    unsafe_allow_html=True,
                )
                st.session_state.entries[idx]["op"] = "+"

        with row[2]:
            st.session_state.entries[idx]["text"] = st.text_area(
                f"文章 {idx + 1}",
                value=entry["text"],
                key=f"text_{entry_id}",
                height=84,
                placeholder="文章を入力してください",
                label_visibility="collapsed",
            )

        if show_move_buttons:
            with row[3]:
                st.button(
                    "↑",
                    key=f"move_up_{entry_id}",
                    on_click=move_group,
                    args=(anchor, "up"),
                    use_container_width=True,
                )

            with row[4]:
                st.button(
                    "↓",
                    key=f"move_down_{entry_id}",
                    on_click=move_group,
                    args=(anchor, "down"),
                    use_container_width=True,
                )
        else:
            with row[3]:
                st.markdown("")
            with row[4]:
                st.markdown("")

        with row[5]:
            st.button(
                "削除",
                key=f"remove_{entry_id}",
                on_click=remove_entry,
                args=(idx,),
                use_container_width=True,
            )

    container = st.container(border=True)
    with container:
        idx = 0
        while idx < len(st.session_state.entries):
            entry = st.session_state.entries[idx]
            if entry["kind"] == "normal":
                render_entry_row(idx)
                idx += 1
                continue

            block_start = idx
            block_id = st.session_state.entries[block_start]["block_id"]
            while (
                idx < len(st.session_state.entries)
                and st.session_state.entries[idx]["kind"] == "composed"
                and st.session_state.entries[idx].get("block_id") == block_id
            ):
                idx += 1
            block_end = idx

            st.caption(f"足し引き文章ブロック {block_id}")
            block = st.container(border=True)
            with block:
                header_cols = st.columns([8.2, 0.9, 0.9])
                with header_cols[0]:
                    st.caption("この範囲の文章が1つの合成ベクトルにまとめられます。")
                with header_cols[1]:
                    st.button(
                        "↑",
                        key=f"move_up_block_{block_id}",
                        on_click=move_group,
                        args=(block_start, "up"),
                        use_container_width=True,
                    )
                with header_cols[2]:
                    st.button(
                        "↓",
                        key=f"move_down_block_{block_id}",
                        on_click=move_group,
                        args=(block_start, "down"),
                        use_container_width=True,
                    )
                current_normalize = bool(st.session_state.entries[block_start].get("normalize", False))
                normalize_value = st.toggle(
                    "項数で割る",
                    value=current_normalize,
                    key=f"normalize_block_{block_id}",
                    help="オンにすると、足し引き後のベクトルをこのブロックの項数で割ります。",
                )
                for composed_idx in range(block_start, block_end):
                    st.session_state.entries[composed_idx]["normalize"] = normalize_value
                for composed_idx in range(block_start, block_end):
                    render_entry_row(composed_idx, show_move_buttons=False, move_anchor=block_start)
                _, add_col = st.columns([8.8, 1.2])
                with add_col:
                    st.button(
                        "追加",
                        key=f"add_block_{block_id}",
                        on_click=add_entry_to_block,
                        args=(block_id,),
                        use_container_width=True,
                    )

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        add_row = st.columns([2.2, 1.2, 6.6])
        with add_row[0]:
            st.selectbox(
                "追加種類",
                options=["normal", "composed"],
                format_func=lambda value: "通常文章を追加" if value == "normal" else "足し引きブロックを追加",
                key="new_entry_kind",
                label_visibility="collapsed",
            )
        with add_row[1]:
            st.button("追加", on_click=add_entry, use_container_width=True)
        with add_row[2]:
            st.caption("足し引きブロックを追加すると、新しい合成ベクトルのまとまりが作られます。")


def collect_inputs() -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    idx = 0

    while idx < len(st.session_state.entries):
        entry = st.session_state.entries[idx]
        text = entry["text"].strip()

        if entry["kind"] == "normal":
            if text:
                groups.append({"kind": "normal", "text": text})
            idx += 1
            continue

        block_id = entry["block_id"]
        block_entries: list[dict[str, str]] = []
        normalize = bool(entry.get("normalize", False))
        while (
            idx < len(st.session_state.entries)
            and st.session_state.entries[idx]["kind"] == "composed"
            and st.session_state.entries[idx].get("block_id") == block_id
        ):
            current = st.session_state.entries[idx]
            current_text = current["text"].strip()
            if current_text:
                block_entries.append({"op": current["op"], "text": current_text})
            idx += 1

        if block_entries:
            groups.append(
                {
                    "kind": "composed",
                    "block_id": block_id,
                    "normalize": normalize,
                    "entries": block_entries,
                }
            )

    return groups


def render_results(model: SentenceTransformer, groups: list[dict[str, object]]) -> None:
    texts_to_embed: list[str] = []
    for group in groups:
        if group["kind"] == "normal":
            texts_to_embed.append(group["text"])
        else:
            texts_to_embed.extend(entry["text"] for entry in group["entries"])

    with st.spinner(f"{MODEL_NAME} で埋め込みを計算しています..."):
        embedded = embed_texts(model, texts_to_embed)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    expanded_vectors: list[np.ndarray] = []
    expanded_labels: list[str] = []
    comparison_vectors: list[np.ndarray] = []
    comparison_labels: list[str] = []

    offset = 0
    for group in groups:
        if group["kind"] == "normal":
            vector = embedded[offset]
            offset += 1
            label = shorten_label(group["text"])
            expanded_vectors.append(vector)
            expanded_labels.append(label)
            comparison_vectors.append(vector)
            comparison_labels.append(label)
            continue

        block_entries = group["entries"]
        block_len = len(block_entries)
        block_vectors = [vector for vector in embedded[offset : offset + block_len]]
        offset += block_len

        signed_vectors = [
            vector if entry["op"] == "+" else -vector
            for entry, vector in zip(block_entries, block_vectors, strict=True)
        ]
        combined_vector = np.sum(np.stack(signed_vectors, axis=0), axis=0)
        if group.get("normalize", False):
            combined_vector = combined_vector / block_len
        expression_label = compose_expression(block_entries, max_len=18)

        expanded_vectors.extend(block_vectors)
        expanded_labels.extend(
            [f"{entry['op']}「{shorten_label(entry['text'])}」" for entry in block_entries]
        )
        expanded_vectors.append(combined_vector)
        expanded_labels.append(expression_label)

        comparison_vectors.append(combined_vector)
        comparison_labels.append(expression_label)

    if comparison_vectors:
        st.caption("比較表示")
        render_heatmap_block(comparison_vectors, comparison_labels)
        st.caption("コサイン類似度")
        render_cosine_similarity(comparison_vectors, comparison_labels)

    if expanded_vectors:
        st.caption("全体表示")
        render_heatmap_block(expanded_vectors, expanded_labels)


def main() -> None:
    init_state()

    st.title("Embedding Heatmap Viewer")
    st.caption(f"モデル: `{MODEL_NAME}` / CPU")

    try:
        with st.spinner("埋め込みモデルを読み込んでいます..."):
            model = load_model()
    except Exception as exc:
        st.error("モデルの読み込みに失敗しました。初回はモデルのダウンロードが必要です。")
        st.exception(exc)
        return

    render_entry_editor()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        submitted = st.button("ベクトルを計算してヒートマップを生成", use_container_width=True)

    if not submitted:
        return

    groups = collect_inputs()
    if not groups:
        st.warning("少なくとも1つは文章を入力してください。")
        return

    render_results(model, groups)


if __name__ == "__main__":
    main()
