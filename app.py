import streamlit as st
import asyncio
import nest_asyncio
import logging
import numpy as np
import pickle
import pandas as pd
import ast
import os

# 이벤트 루프 세팅
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nest_asyncio.apply()

import laftel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --------------------------
# 데이터 불러오기
# --------------------------
combined_df = pd.read_csv("./combined_df_plus.csv")
with open("./ani_emb_gray_person_plus.pkl", "rb") as f:
    cosine_similarities = pickle.load(f)


def get_index_by_id(df, anime_id):
    result = df.index[df["anime_id"] == anime_id]
    return result[0] if len(result) > 0 else None

def replace_last_colon_except_after_E(p: str) -> str:
    # "E:" 위치는 건드리지 않도록 무시
    # 마지막 콜론 위치 찾기 (단, 'E:'가 바로 앞에 붙은 콜론 제외)
    
    # 마지막 콜론 위치
    last_idx = p.rfind(':')
    if last_idx == -1:
        return p  # 콜론 없으면 그대로

    # 바로 앞 문자가 'E'인지 확인
    if last_idx > 0 and p[last_idx - 1] == 'E':
        # 만약 마지막 콜론이 'E:'라면, 그 바로 앞 콜론 위치를 찾아야 함
        # 'E:' 바로 옆 콜론은 건너뛰고 그 이전 콜론 찾기
        second_last_idx = p.rfind(':', 0, last_idx)
        if second_last_idx == -1:
            # 두 번째 콜론 없으면 바꿀 콜론 없음
            return p
        else:
            # 두 번째 콜론 위치를 바꿔줌
            return p[:second_last_idx] + '_' + p[second_last_idx + 1:]
    else:
        # 마지막 콜론이 'E:' 옆이 아니면 그냥 마지막 콜론 바꿈
        return p[:last_idx] + '_' + p[last_idx + 1:]


def fix_windows_path(p):
    p = p.replace('/content/drive/My Drive/', './')
    p = replace_last_colon_except_after_E(p)  # Windows 파일 시스템에서 ':'는 금지됨, 유니코드 콜론으로 대체
    p = os.path.normpath(p)  # 경로 정규화
    return p


def main():
    st.title("Anime Recommendation WebApp")

    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "anime_options" not in st.session_state:
        st.session_state.anime_options = []
    if "recommended_list" not in st.session_state:
        st.session_state.recommended_list = []
    if "recommended_images" not in st.session_state:
        st.session_state.recommended_images = []
    if "image_indices" not in st.session_state:
        st.session_state.image_indices = {}

    if "target_images" not in st.session_state:
        st.session_state.target_images = []
    if "target_index" not in st.session_state:
        st.session_state.target_index = 0

    query = st.text_input("검색할 애니 이름을 입력하세요:", "")
    if st.button("검색"):
        logging.info("검색 버튼 클릭!")
        results = laftel.sync.searchAnime(query)

        st.session_state.search_results = results
        if len(results) == 0:
            st.write("검색 결과가 없습니다.")
            st.session_state.anime_options = []
        else:
            options = [f"{i}. {r.name} (ID={r.id})" for i, r in enumerate(results)]
            st.session_state.anime_options = options
            st.write("검색 완료! 아래 selectbox에서 애니를 선택하세요.")

    selected_option = None
    if st.session_state.anime_options:
        selected_option = st.selectbox("선택할 애니를 고르세요:", st.session_state.anime_options)

    if st.session_state.anime_options:
        if st.button("추천보기"):
            if st.session_state.search_results and selected_option is not None:
                logging.info("추천보기 버튼 클릭!")

                selected_index = int(selected_option.split(".")[0])
                selected_anime = st.session_state.search_results[selected_index]
                anime_id = selected_anime.id
                target_index = get_index_by_id(combined_df, anime_id)
                if target_index is None:
                    st.write("데이터프레임에서 해당 anime_id를 찾을 수 없습니다.")
                    return

                similarity_row = cosine_similarities[target_index]
                most_similar_indices = np.argsort(similarity_row)[::-1]

                target_row = combined_df.iloc[target_index]
                st.write(f"**선택한 애니메이션**: {target_row['anime_name']} (id={target_row['anime_id']})")

                # 타겟 이미지 경로
                target_pic_str = target_row["thumbnail_path_full"]
                target_pic_paths = ast.literal_eval(target_pic_str)
                st.session_state.target_images = [fix_windows_path(p) for p in target_pic_paths]
                st.session_state.target_index = 0

                recommended = []
                total_picture = []
                rank_count = 0
                for idx in most_similar_indices:
                    if idx == target_index:
                        continue
                    sim_score = similarity_row[idx]
                    rec_name = combined_df.iloc[idx]["anime_name"]
                    rec_pic_str = combined_df.iloc[idx]["thumbnail_path_full"]
                    rec_pic_paths = ast.literal_eval(rec_pic_str)
                    one_ani_picture = [fix_windows_path(p) for p in rec_pic_paths]

                    total_picture.append(one_ani_picture)
                    recommended.append((rec_name, sim_score))
                    rank_count += 1
                    if rank_count >= 5:
                        break

                st.session_state.recommended_list = recommended
                st.session_state.recommended_images = total_picture
                st.session_state.image_indices = {}

    # 타겟 애니 스틸컷 보기
    if st.session_state.target_images:
        colA, colB, colC = st.columns([1, 6, 1])
        with colA:
            if st.button("⬅️", key="prev_target"):
                st.session_state.target_index = (
                    st.session_state.target_index - 1
                ) % len(st.session_state.target_images)
        with colC:
            if st.button("➡️", key="next_target"):
                st.session_state.target_index = (
                    st.session_state.target_index + 1
                ) % len(st.session_state.target_images)

        st.image(
            st.session_state.target_images[st.session_state.target_index],
            caption="선택한 애니 스틸컷",
            use_container_width=True
        )

    # 추천 결과 및 이미지 넘기기
    if st.session_state.recommended_list:
        st.write("**비슷한 애니 Top 5**")

        for i, (name, score) in enumerate(st.session_state.recommended_list, start=1):
            st.write(f"{i}. {name} (유사도: {score:.4f})")

            image_paths = st.session_state.recommended_images[i-1]
            key = f"img_idx_{i}"

            if key not in st.session_state.image_indices:
                st.session_state.image_indices[key] = 0

            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                if st.button("⬅️", key=f"prev_{key}"):
                    st.session_state.image_indices[key] = (
                        st.session_state.image_indices[key] - 1
                    ) % len(image_paths)
            with col3:
                if st.button("➡️", key=f"next_{key}"):
                    st.session_state.image_indices[key] = (
                        st.session_state.image_indices[key] + 1
                    ) % len(image_paths)

            current_idx = st.session_state.image_indices[key]
            st.image(image_paths[current_idx], caption="스틸컷", use_container_width=True)


if __name__ == "__main__":
    main()
