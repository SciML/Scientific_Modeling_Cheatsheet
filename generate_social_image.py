#!/usr/bin/env python3
"""
Generate a social media image for the Scientific Modeling Cheatsheet
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Image dimensions (1200x630 is optimal for Twitter/Facebook/LinkedIn)
WIDTH = 1200
HEIGHT = 630

# Colors
BACKGROUND = "#1a1a2e"  # Dark blue-grey
ACCENT_1 = "#16213e"    # Darker blue
ACCENT_2 = "#0f3460"    # Medium blue
MATLAB_COLOR = "#e97451"  # Orange-red for MATLAB
PYTHON_COLOR = "#3776ab"  # Blue for Python
JULIA_COLOR = "#9558b2"   # Purple for Julia
TEXT_COLOR = "#eaeaea"    # Light grey for text
TITLE_COLOR = "#ffffff"   # White for title

def create_social_image(output_path="social_image.png"):
    """Create the social media image"""

    # Create image with background
    img = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(img)

    # Try to use better fonts if available, otherwise use default
    try:
        # Try common font locations
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        lang_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        lang_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw decorative rectangles
    draw.rectangle([0, 0, 20, HEIGHT], fill=MATLAB_COLOR)
    draw.rectangle([WIDTH-20, 0, WIDTH, HEIGHT], fill=JULIA_COLOR)

    # Draw title
    title = "Scientific Modeling"
    title2 = "Cheatsheet"

    # Get text bounding boxes for centering
    bbox1 = draw.textbbox((0, 0), title, font=title_font)
    bbox2 = draw.textbbox((0, 0), title2, font=title_font)

    title_width1 = bbox1[2] - bbox1[0]
    title_width2 = bbox2[2] - bbox2[0]

    # Draw title (centered)
    draw.text(((WIDTH - title_width1) // 2, 100), title, fill=TITLE_COLOR, font=title_font)
    draw.text(((WIDTH - title_width2) // 2, 190), title2, fill=TITLE_COLOR, font=title_font)

    # Draw subtitle
    subtitle = "Quick Reference Guide"
    bbox_sub = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = bbox_sub[2] - bbox_sub[0]
    draw.text(((WIDTH - subtitle_width) // 2, 300), subtitle, fill=TEXT_COLOR, font=subtitle_font)

    # Draw language boxes
    y_start = 400
    box_width = 280
    box_height = 100
    spacing = 40
    total_width = 3 * box_width + 2 * spacing
    x_start = (WIDTH - total_width) // 2

    languages = [
        ("MATLAB", MATLAB_COLOR),
        ("Python", PYTHON_COLOR),
        ("Julia", JULIA_COLOR)
    ]

    for i, (lang, color) in enumerate(languages):
        x = x_start + i * (box_width + spacing)

        # Draw rounded rectangle (simulate with rectangle)
        draw.rectangle([x, y_start, x + box_width, y_start + box_height],
                      fill=color, outline=TEXT_COLOR, width=3)

        # Draw language name
        bbox_lang = draw.textbbox((0, 0), lang, font=lang_font)
        lang_width = bbox_lang[2] - bbox_lang[0]
        lang_height = bbox_lang[3] - bbox_lang[1]

        text_x = x + (box_width - lang_width) // 2
        text_y = y_start + (box_height - lang_height) // 2 - 10

        draw.text((text_x, text_y), lang, fill=TITLE_COLOR, font=lang_font)

    # Draw footer
    footer = "github.com/SciML/Scientific_Modeling_Cheatsheet"
    bbox_footer = draw.textbbox((0, 0), footer, font=small_font)
    footer_width = bbox_footer[2] - bbox_footer[0]
    draw.text(((WIDTH - footer_width) // 2, HEIGHT - 60), footer,
             fill=TEXT_COLOR, font=small_font)

    # Save image
    img.save(output_path, 'PNG', optimize=True)
    print(f"Social media image saved to: {output_path}")
    print(f"Image dimensions: {WIDTH}x{HEIGHT}")

if __name__ == "__main__":
    create_social_image()
