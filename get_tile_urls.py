#!/usr/bin/env python3
from generate_maps import tile_dir, lon_lat_to_x_y
import os
import argparse
import json


def get_tile_urls(min_x, max_x, min_y, max_y, zoom, tileserver):
    urls = []
    os.makedirs(tile_dir, exist_ok=True)
    for x in range(min_x, max_x):
        img_dir = os.path.join(tile_dir, str(zoom), str(x))
        os.makedirs(img_dir, exist_ok=True)
        for y in range(min_y, max_y):
            img_file = os.path.join(img_dir, f'{y}.png')
            if os.path.exists(img_file) and os.path.getsize(img_file) > 0:
                continue
            img_url = f'{tileserver}/{zoom}/{x}/{y}.png'
            urls.append(img_url)
    return urls


def main(bounding_boxes_latlon_json, output_file, zoom_levels, tileserver):
    # Download all the tiles
    with open(bounding_boxes_latlon_json, 'r') as f:
        bounding_boxes_latlon = json.load(f)
    if os.path.exists(output_file):
        os.remove(output_file)
    for zoom in zoom_levels:
        bounding_boxes_tiles = {}
        for name, coords in bounding_boxes_latlon.items():
            start_x, start_y = lon_lat_to_x_y(coords['start_lon'], coords['start_lat'], zoom)
            end_x, end_y = lon_lat_to_x_y(coords['end_lon'], coords['end_lat'], zoom)
            bounding_boxes_tiles[name] = {
                'start_x': start_x,
                'start_y': start_y,
                'end_x': end_x + 1,
                'end_y': end_y + 1,
            }

        for name, coords in bounding_boxes_tiles.items():
            urls = get_tile_urls(coords['start_x'], coords['end_x'], coords['start_y'], coords['end_y'], zoom, tileserver)
            if urls:
                with open(output_file, 'a') as f:
                    f.write('\n'.join(urls) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate tile URLs for specified bounding boxes.")
    parser.add_argument(
        '--input',
        help='Input JSON file with bounding boxes for one or more regions.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--output',
        help='Output file to write tile URLs (default: tile_urls.txt)',
        default='tile_urls.txt',
        type=str
    )
    parser.add_argument(
        '--zoom',
        help='Comma-separated list of integer zoom levels.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--tileserver',
        help='Base URL of an OSM tile server. Default is `http://localhost:8080/tile` (assumes you are running your own). Not used if all tiles are already present.',
        default='http://localhost:8080/tile',
        type=str
    )
    args = parser.parse_args()
    assert os.path.isfile(args.input), f'Error: {args.input} does not exist.'
    assert not os.path.isfile(args.output), f'Error: {args.output} already exists.'
    zoom_levels = args.zoom.split(',')
    zoom_levels = [int(zoom) for zoom in zoom_levels]

    main(args.input, args.output, zoom_levels, args.tileserver)