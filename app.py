#!/usr/bin/env python3
"""
Simple CLI runner for the line/character segmentation pipeline.
Saves intermediate and final outputs to the `outputs/` directory.
"""
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import io
import zipfile


def ensure_out(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def save_gray(img, path):
    cv2.imwrite(path, img)


def save_color(img, path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img)


def plot_and_save_projection(proj, path, title="Projection"):
    plt.figure(figsize=(10, 4))
    plt.plot(proj)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Sum')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def process_image(input_path, outdir='outputs'):
    """Process an image path and save outputs into outdir.

    Returns a dict of produced file paths.
    """
    ensure_out(outdir)

    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f'Failed to read image: {input_path}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    results = {}

    # 1. Canny
    imgcanny = cv2.Canny(gray, 150, 200)
    results['canny'] = os.path.join(outdir, 'canny.png')
    save_gray(imgcanny, results['canny'])

    # 2. Dilate
    kernel = np.ones((1, 2), np.uint8)
    imgdilate = cv2.dilate(imgcanny, kernel, iterations=2)
    results['dilate'] = os.path.join(outdir, 'dilate.png')
    save_gray(imgdilate, results['dilate'])

    # 3. Threshold to binary
    _, imgbin = cv2.threshold(imgdilate, 127, 255, cv2.THRESH_BINARY)
    results['binary'] = os.path.join(outdir, 'binary.png')
    save_gray(imgbin, results['binary'])

    # 4. Contour filtering
    img_for_contours = imgbin.copy()
    contours_info = cv2.findContours(img_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    cleaned_contour_image = np.zeros_like(imgbin)

    min_area = 1
    max_area = 5000
    min_width = 1
    max_width = 50
    min_height = 1
    max_height = 95

    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if (min_area < area < max_area) and (min_width < w < max_width) and (min_height < h < max_height):
            filtered_contours.append(contour)

    cv2.drawContours(cleaned_contour_image, filtered_contours, -1, 255, thickness=2)
    results['cleaned_contours'] = os.path.join(outdir, 'cleaned_contours.png')
    save_gray(cleaned_contour_image, results['cleaned_contours'])

    # 5. Close small gaps horizontally
    LINE_CLOSING_KERNEL_WIDTH = 50
    temp_opening_kernel = np.ones((1, LINE_CLOSING_KERNEL_WIDTH), np.uint8)
    temp_cleaned_image_for_projection = cv2.morphologyEx(cleaned_contour_image, cv2.MORPH_CLOSE, temp_opening_kernel)
    results['closed'] = os.path.join(outdir, 'closed.png')
    save_gray(temp_cleaned_image_for_projection, results['closed'])

    # 6. Horizontal projection
    horizontal_projection = np.sum(temp_cleaned_image_for_projection, axis=1)
    results['horizontal_projection'] = os.path.join(outdir, 'horizontal_projection.png')
    plot_and_save_projection(horizontal_projection, results['horizontal_projection'], title='Horizontal Projection')

    # 7. Line segmentation
    LINE_SEGMENTATION_THRESHOLD_MULTIPLIER = 1.1
    non_zero = horizontal_projection[horizontal_projection > 0]
    if non_zero.size > 0:
        threshold_projection = np.mean(non_zero) * LINE_SEGMENTATION_THRESHOLD_MULTIPLIER
    else:
        threshold_projection = 0

    line_starts, line_ends = [], []
    is_in_line = False
    for i, value in enumerate(horizontal_projection):
        if value > threshold_projection and not is_in_line:
            line_starts.append(i)
            is_in_line = True
        elif value <= threshold_projection and is_in_line:
            line_ends.append(i - 1)
            is_in_line = False
    if is_in_line:
        line_ends.append(len(horizontal_projection) - 1)

    results['lines'] = []
    separated_lines = []
    separated_lines_original = []
    for i in range(len(line_starts)):
        s = line_starts[i]
        e = line_ends[i]
        line_img = cleaned_contour_image[s:e+1, :]
        separated_lines.append(line_img)
        line_img_orig = img[s:e+1, :]
        separated_lines_original.append(line_img_orig)
        # save per-line
        p_bin = os.path.join(outdir, f'line_{i+1:02d}_bin.png')
        p_orig = os.path.join(outdir, f'line_{i+1:02d}_orig.png')
        save_gray(line_img, p_bin)
        cv2.imwrite(p_orig, line_img_orig)
        results['lines'].append({'bin': p_bin, 'orig': p_orig, 'start': s, 'end': e})

    # 8. Character/word segmentation and display assembly
    MIN_SEGMENT_WIDTH = 5
    CHAR_SEGMENTATION_THRESHOLD_MULTIPLIER = 0.05

    display_segments = []
    display_segments_original = []

    for line_idx, line_img in enumerate(separated_lines):
        if line_img.size == 0 or line_img.shape[1] == 0:
            continue
        vertical_projection = np.sum(line_img, axis=0)
        nz = vertical_projection[vertical_projection > 0]
        if nz.size > 0:
            threshold_for_segmentation = np.mean(nz) * CHAR_SEGMENTATION_THRESHOLD_MULTIPLIER
        else:
            threshold_for_segmentation = 0

        segments = []
        is_seg = False
        start_col = 0
        for col_idx, val in enumerate(vertical_projection):
            if val > threshold_for_segmentation and not is_seg:
                start_col = col_idx
                is_seg = True
            elif val <= threshold_for_segmentation and is_seg:
                end_col = col_idx - 1
                if (end_col - start_col + 1) >= MIN_SEGMENT_WIDTH:
                    seg = line_img[:, start_col:end_col+1]
                    segments.append(seg)
                is_seg = False
        if is_seg:
            end_col = len(vertical_projection) - 1
            if (end_col - start_col + 1) >= MIN_SEGMENT_WIDTH:
                segments.append(line_img[:, start_col:end_col+1])

        if segments:
            sep_w = 10
            separator = np.full((line_img.shape[0], sep_w), 255, dtype=np.uint8)
            combined = []
            for j, seg in enumerate(segments):
                combined.append(seg)
                if j < len(segments) - 1:
                    combined.append(separator)
            display_segments.append(np.concatenate(combined, axis=1))

        # original color segments
        orig_line = separated_lines_original[line_idx]
        if segments and orig_line.size != 0:
            combined_color = []
            sep_color = np.full((orig_line.shape[0], sep_w, 3), 255, dtype=np.uint8)
            col_offset = 0
            for j, seg in enumerate(segments):
                ec = seg.shape[1]
                color_seg = orig_line[:, col_offset:col_offset+ec]
                combined_color.append(color_seg)
                if j < len(segments) - 1:
                    combined_color.append(sep_color)
                col_offset += ec
            try:
                display_segments_original.append(np.concatenate(combined_color, axis=1))
            except Exception:
                pass

    if display_segments:
        max_w = max(img.shape[1] for img in display_segments)
        final_imgs = []
        line_sep = np.full((20, max_w), 255, dtype=np.uint8)
        for i, img_disp in enumerate(display_segments):
            if img_disp.shape[1] < max_w:
                pad = np.full((img_disp.shape[0], max_w - img_disp.shape[1]), 255, dtype=np.uint8)
                img_disp = np.concatenate((img_disp, pad), axis=1)
            final_imgs.append(img_disp)
            if i < len(display_segments) - 1:
                final_imgs.append(line_sep)
        combined_all_segments_image = np.concatenate(final_imgs, axis=0)
        results['combined_segments'] = os.path.join(outdir, 'combined_segments.png')
        save_gray(combined_all_segments_image, results['combined_segments'])
    else:
        results['combined_segments'] = None

    if display_segments_original:
        max_w = max(img.shape[1] for img in display_segments_original)
        final_imgs = []
        line_sep_color = np.full((20, max_w, 3), 255, dtype=np.uint8)
        for i, img_disp in enumerate(display_segments_original):
            if img_disp.shape[1] < max_w:
                pad = np.full((img_disp.shape[0], max_w - img_disp.shape[1], 3), 255, dtype=np.uint8)
                img_disp = np.concatenate((img_disp, pad), axis=1)
            final_imgs.append(img_disp)
            if i < len(display_segments_original) - 1:
                final_imgs.append(line_sep_color)
        combined_all_segments_image_original = np.concatenate(final_imgs, axis=0)
        results['combined_segments_original'] = os.path.join(outdir, 'combined_segments_original.png')
        cv2.imwrite(results['combined_segments_original'], combined_all_segments_image_original)
    else:
        results['combined_segments_original'] = None

    return results


def main():
    p = argparse.ArgumentParser(description='Run segmentation pipeline and save results to outputs/')
    p.add_argument('--input', '-i', help='Input image path (optional)')
    p.add_argument('--out', '-o', default='outputs', help='Output folder')
    args = p.parse_args()

    input_path = args.input
    if not input_path:
        # fallback to workspace image
        fallback = os.path.join(os.getcwd(), 'image8c.png')
        if os.path.exists(fallback):
            print(f'No input given — using fallback: {fallback}')
            input_path = fallback
        else:
            raise SystemExit('No input provided and fallback image8c.png not found in workspace')

    if not os.path.exists(input_path):
        raise SystemExit(f'Input not found: {input_path}')

    outdir = args.out
    results = process_image(input_path, outdir=outdir)
    print('Results:', results)


def _streamlit_app():
    import streamlit as st

    st.title('Line Segmentation — Upload Image')
    uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'bmp', 'tif'])

    outdir = os.path.join(os.getcwd(), 'outputs')
    ensure_out(outdir)

    if uploaded is not None:
        # save uploaded to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        st.image(uploaded, caption='Uploaded image', use_column_width=True)
        with st.spinner('Processing...'):
            results = process_image(tmp_path, outdir=outdir)

        st.success('Processing finished')

        # show key outputs
        if results.get('combined_segments'):
            st.subheader('Detected line segments (grayscale)')
            st.image(results['combined_segments'], use_column_width=True)
            with open(results['combined_segments'], 'rb') as f:
                st.download_button('Download grayscale segments', f.read(), file_name='combined_segments.png')

        if results.get('combined_segments_original'):
            st.subheader('Detected line segments (original color)')
            st.image(results['combined_segments_original'], use_column_width=True)
            with open(results['combined_segments_original'], 'rb') as f:
                st.download_button('Download color segments', f.read(), file_name='combined_segments_original.png')

        # allow download of a zip of outputs
        if st.button('Download all outputs (zip)'):
            zip_path = os.path.join(outdir, 'outputs.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fn in os.listdir(outdir):
                    if fn == 'outputs.zip':
                        continue
                    zf.write(os.path.join(outdir, fn), arcname=fn)
            with open(zip_path, 'rb') as f:
                st.download_button('Download ZIP', f.read(), file_name='outputs.zip')


if __name__ == '__main__':
    try:
        import streamlit as st  # type: ignore
    except Exception:
        main()
    else:
        _streamlit_app()