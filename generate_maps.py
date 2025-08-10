#!/usr/bin/env python3
import requests
import json
import matplotlib.pyplot as plt
import os
import math
import argparse
import glob
import pandas as pd
import sys


# Define constants:
# data directories
route_dir = 'data/routes'
tile_dir = 'data/tiles'
# image/render constants
dpi = 300
a4_long=11.69
a4_short=8.27
max_tiles_long = 9
max_tiles_short = 5
route_base_url = 'https://api.openrouteservice.org/v2/directions/driving-car'
route_api_key_file = '.openrouteapikey'


# Define functions to convert between OSM and geo coordinates
def lon_lat_to_x_y(lon, lat, zoom):
    assert isinstance(zoom, int) and zoom > 0, 'Error: zoom not valid.'
    assert isinstance(lon, (int, float)) and lon >= -180 and lon <= 180, 'Error: lon not valid.'
    assert isinstance(lat, (int, float)) and lat >= -90 and lat <= 90, 'Error: lat not valid.'
    n = 2 ** int(zoom)
    x = math.radians(lon)
    x = (1 + (x / math.pi)) / 2
    x = math.floor(n * x)

    y = math.radians(lat)
    y = math.asinh(math.tan(y))
    y = (1 - (y / math.pi)) / 2
    y = math.floor(n * y)

    return (x, y)


def x_y_to_lon_lat(x, y, zoom):
    assert isinstance(zoom, int) and zoom > 0, 'Error: zoom not valid.'
    assert isinstance(x, int) and x >= 0, 'Error: x not valid.'
    assert isinstance(y, int) and y >= 0, 'Error: x not valid.'
    n = 2 ** int(zoom)
    lon = math.degrees(((2 * x / n) - 1) * math.pi)
    lat = math.degrees(math.atan(math.sinh((1 - (2 * y / n)) * math.pi)))

    return (lon, lat)


# Download route
def get_route(start_lon, start_lat, end_lon, end_lat, apikey):
    os.makedirs(route_dir, exist_ok=True)
    start_lon_s = str(start_lon).replace('-', '_')
    start_lat_s = str(start_lat).replace('-', '_')
    end_lon_s = str(end_lon).replace('-', '_')
    end_lat_s = str(end_lat).replace('-', '_')
    datafile = os.path.join(route_dir, f'data.{start_lon_s}.{start_lat_s}.{end_lon_s}.{end_lat_s}.json')
    if not os.path.isfile(datafile):
        start_lonlat = f'{start_lon:.5f},{start_lat:.5f}'

        end_lonlat = f'{end_lon:.5f},{end_lat:.5f}'

        url = f'{route_base_url}?api_key={apikey}&start={start_lonlat}&end={end_lonlat}'

        headers = {
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        }

        call = requests.get(url, headers=headers)

        with open(datafile, 'w') as f:
            f.write(call.text)

    with open(datafile, 'r') as f:
        data = json.load(f)

    if 'features' not in data:
        os.remove(datafile)
        sys.exit(1)

    route_coords = data['features'][0]['geometry']['coordinates']
    route_x_coords = [c[0] for c in route_coords]
    route_y_coords = [c[1] for c in route_coords]

    return (route_x_coords, route_y_coords)


# Download tiles
def get_tiles(min_x, max_x, min_y, max_y, zoom, tileserver):
    os.makedirs(tile_dir, exist_ok=True)
    for x in range(min_x, max_x):
        img_dir = os.path.join(tile_dir, str(zoom), str(x))
        os.makedirs(img_dir, exist_ok=True)
        for y in range(min_y, max_y):
            img_file = os.path.join(img_dir, f'{y}.png')
            if os.path.exists(img_file):
                continue
            img_data = requests.get(f'{tileserver}/{zoom}/{x}/{y}.png').content
            with open(img_file, 'wb') as f:
                f.write(img_data)


# Plot tiles
def plot_tiles(ax, min_x, max_x, min_y, max_y, zoom):
    for x in range(min_x, max_x):
        img_dir = os.path.join(tile_dir, str(zoom), str(x))
        for y in range(min_y, max_y):
            lon_x, lat_y = x_y_to_lon_lat(x, y, zoom)
            lon_x1, lat_y1 = x_y_to_lon_lat(x + 1, y + 1, zoom)
            img_file = os.path.join(img_dir, f'{y}.png')
            img = plt.imread(img_file)
            ax.imshow(img, extent=[lon_x, lon_x1, lat_y1, lat_y])
    return ax


# Plot all tiles
def plot_all_tiles(route_coords_min_max, zoom):
    fig, ax = plt.subplots()
    for name, coords in route_coords_min_max.items():
        ax = plot_tiles(ax, coords['min_x'], coords['max_x'], coords['min_y'], coords['max_y'], zoom)
    return fig, ax


# Get min max route coords for all routes
def get_min_max_route_coords(route_coords_min_max):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    first = True

    for coords in route_coords_min_max.values():
        if first:
            min_x = coords['min_x']
            max_x = coords['max_x']
            min_y = coords['min_y']
            max_y = coords['max_y']
            first = False
            continue

        if coords['min_x'] < min_x:
            min_x = coords['min_x']
        if coords['max_x'] > max_x:
            max_x = coords['max_x']
        if coords['min_y'] < min_y:
            min_y = coords['min_y']
        if coords['max_y'] > max_y:
            max_y = coords['max_y']

    return (min_x, max_x, min_y, max_y)


def get_avg_lat_rad(route_coords):
    min_y = 0
    max_y = 0
    first = True
    for coords in route_coords.values():
        next_min_y = min(coords['y'])
        next_max_y = max(coords['y'])
        if first:
            min_y = next_min_y
            max_y = next_max_y
            first = False
            continue
        if next_min_y < min_y:
            min_y = next_min_y
        elif next_max_y > max_y:
            max_y = next_max_y
    return math.radians(abs((max_y + min_y) / 2))


def get_windows(distance, window_size, crop=True, recalculate_last_window=True):
    n_windows = math.ceil(distance / window_size)
    stride = math.ceil(distance / n_windows)
    if crop:
        windows = [
            (i * stride, min(i * stride + window_size, distance))
            for i in range(0, n_windows)
        ]
    else:
        windows = [
            (i * stride, i * stride + window_size)
            for i in range(0, n_windows)
        ]
    
    # Recalculate last window if requested
    if crop and recalculate_last_window:
        i = n_windows - 1
        windows[i] = (distance - window_size, distance)
    return windows


def read_input(input_csv):
    # Read CSV file
    df = pd.read_csv(input_csv)
    # Clean up
    df.columns = [str(c).lower() for c in df.columns]
    # latitude and longitude are required
    lat_col = 'latitude'
    lon_col = 'longitude'
    req_cols = [lat_col, lon_col]
    assert all([c in df.columns for c in req_cols])
    # location is optional
    loc_col = 'location'
    all_cols = [loc_col] + req_cols
    select_cols = all_cols if loc_col in df.columns else req_cols
    df = df[select_cols]
    # Drop rows with missing coordinates
    df.dropna(axis='index', subset=req_cols, how='any', inplace=True)
    # If location is missing, create placeholder sequence
    tmp_loc_col = 'tmp_location'
    df[tmp_loc_col] = [f'stop{i}' for i in range(0, df.shape[0])]
    if loc_col not in df.columns:
        df.rename(columns={tmp_loc_col: loc_col}, inplace=True)
    else:
        df.fillna({loc_col: 'NA'}, inplace=True)
        df[loc_col] = df.apply(lambda x: x[tmp_loc_col] if x[loc_col] == 'NA' else x[loc_col], axis='columns')
        df.drop(columns=tmp_loc_col, inplace=True)
    # Shift coordinates to get end points
    loc_col_2 = 'location2'
    df2 = df.shift(-1).rename(columns={loc_col: loc_col_2, lat_col: 'end_lat', lon_col: 'end_lon'})
    # Rename columns
    df.rename(columns={lat_col: 'start_lat', lon_col: 'start_lon'}, inplace=True)
    # Concatenate
    df = pd.concat([df, df2], axis='columns').iloc[:-1]
    # Combine location columns
    df[loc_col] = df[loc_col] + '_' + df[loc_col_2]
    df.drop(columns=loc_col_2, inplace=True)
    # Set index
    df.set_index('location', inplace=True)
    return df.T.to_dict()


def main(input, prefix, zoom, colours, linewidth, linealpha, route_api_key, tileserver):
    # Define start and end latitudes and longitudes for each route
    with open(input, 'r') as f:
        routes = read_input(input)

    # Get route coordinates
    route_coords = {}
    route_coords_min_max = {}
    for name, coords in routes.items():
        # Get route coordinates
        route_x_coords, route_y_coords = get_route(coords['start_lon'], coords['start_lat'], coords['end_lon'], coords['end_lat'], route_api_key)

        # Get min and max lon and lat values
        min_lon = min(route_x_coords)
        max_lon = max(route_x_coords)
        min_lat = min(route_y_coords)
        max_lat = max(route_y_coords)

        # Get min and max OSM coordinates
        # min_y = max_lat and
        # max_y = min_lat, since
        # OSM y values go from N to S
        min_x, min_y = lon_lat_to_x_y(min_lon, max_lat, zoom)
        max_x, max_y = lon_lat_to_x_y(max_lon, min_lat, zoom)
        max_x1 = max_x + 1
        max_y1 = max_y + 1

        route_coords[name] = {
            'x': route_x_coords,
            'y': route_y_coords,
        }

        route_coords_min_max[name] = {
            'min_x': min_x,
            'max_x': max_x1,
            'min_y': min_y,
            'max_y': max_y1,
        }

    # Get tiles for route
    min_x, max_x, min_y, max_y = get_min_max_route_coords(route_coords_min_max)
    get_tiles(min_x, max_x, min_y, max_y, zoom, tileserver)

    # Figure out how to break up map
    d_x = max_x - min_x
    d_y = max_y - min_y
    orientation = 'LANDSCAPE' if d_x >= d_y else 'PORTRAIT'
    if orientation == 'LANDSCAPE':
        max_tiles_x = max_tiles_long
        max_tiles_y = max_tiles_short
    else:
        max_tiles_x = max_tiles_short
        max_tiles_y = max_tiles_long
    breaks = {'x': [], 'y': []}
    if d_x <= max_tiles_x:
        breaks['x'] = [(min_x, max_x)]
    else:
        windows = get_windows(d_x, max_tiles_x)
        breaks['x'] = [
            (start + min_x, stop + min_x)
            for start, stop in windows
        ]
    if d_y <= max_tiles_y:
        breaks['y'] = [(min_y, max_y)]
    else:
        windows = get_windows(d_y, max_tiles_y)
        breaks['y'] = [
            (start + min_y, stop + min_y)
            for start, stop in windows
        ]

    # Plot maps with coordinates
    fig, ax = plt.subplots()
    ax = plot_tiles(ax, min_x, max_x, min_y, max_y, zoom)
    i = 0
    for route_name, coords in route_coords.items():
        # Plot route
        ax.plot(coords['x'], coords['y'], color=colours[i % len(colours)], linestyle='-', linewidth=linewidth, alpha=linealpha)
        i += 1
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

    # Set aspect ratio
    ax.set_aspect(1 / math.cos(get_avg_lat_rad(route_coords)))

    # Save one giant map
    # plt.savefig(f'map.complete.pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'map.complete.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

    # Set figure size
    ax.set_aspect(1 / math.cos(get_avg_lat_rad(route_coords)))
    if orientation == 'LANDSCAPE':
        fig.set_size_inches(a4_long, a4_short)
        fig.subplots_adjust(left=0.1/a4_long, top=(a4_short - 0.1)/a4_short)
    else:
        fig.set_size_inches(a4_short, a4_long)
        fig.subplots_adjust(left=0.1/a4_short, top=(a4_long - 0.1)/a4_long)

    # If the map is too big, also save multiple broken-up maps
    ix = 0
    for start_x, stop_x in breaks['x']:
        iy = 0
        for start_y, stop_y in breaks['y']:
            start_lon, start_lat = x_y_to_lon_lat(start_x, start_y, zoom)
            stop_lon, stop_lat = x_y_to_lon_lat(stop_x, stop_y, zoom)
            ax.set_xlim(left=start_lon, right=stop_lon)
            ax.set_ylim(bottom=stop_lat, top=start_lat)
            plt.savefig(f'{prefix}.{ix}.{iy}.pdf', dpi=dpi)
            plt.savefig(f'{prefix}.{ix}.{iy}.png', dpi=dpi)
            iy += 1
        ix += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate printable A4-sized maps with marked routes.")
    parser.add_argument(
        '--input',
        help='Input CSV file with latitude and longitude coordinates for each stop in the route to plot.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--zoom',
        help='Zoom level for maps.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--name',
        help='Name of the map. Defaults to `map`.',
        default='map',
        type=str
    )
    parser.add_argument(
        '--outdir',
        help='Output directory. Defaults to `output`.',
        default='output',
        type=str
    )
    parser.add_argument(
        '--colours',
        help='File containing colours to be used for plotting routes, one colour per line. Colours are used in sequence and cyclically. Defaults to `colours.txt`',
        default='colours.txt',
        type=str
    )
    parser.add_argument(
        '--linewidth',
        help='Line width for plotting route. Default: 1',
        default=1,
        type=float
    )
    parser.add_argument(
        '--linealpha',
        help='Line alpha for plotting route. Default: 1',
        default=1,
        type=float
    )
    parser.add_argument(
        '--tileserver',
        help='Base URL of an OSM tile server. Default is `http://localhost:8080/tile` (assumes you are running your own). Not used if all tiles are already present.',
        default='http://localhost:8080/tile',
        type=str
    )
    args = parser.parse_args()
    assert os.path.isfile(args.input), f'Error: {args.input} does not exist.'
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    assert os.path.isdir(args.outdir), f'Error: File with name `{args.outdir}` already exists. Please choose a different output directory name.'
    map_prefix = os.path.join(args.outdir, f'{args.name}')
    existing_maps = glob.glob(f'{map_prefix}*')
    assert not existing_maps, f'Error: Files with prefix {map_prefix} already exist. Please remove them or choose a different output directory or map name.'
    assert os.path.isfile(args.colours), f'Error: Colours file {args.colours} does not exist.'

    with open(args.colours, 'r') as f:
        colours = f.readlines()
        colours = [str(c).strip() for c in colours]

    assert args.linewidth > 0, f'Error: invalid line width supplied: {args.linewidth}'
    assert args.linealpha >= 0 and args.linealpha <= 1, f'Error: invalid alpha supplied: {args.linealpha}'

    with open(route_api_key_file, 'r') as f:
        route_api_key = f.readline().strip()

    main(input=args.input, prefix=map_prefix, zoom=args.zoom, colours=colours, linewidth=args.linewidth, linealpha=args.linealpha, route_api_key=route_api_key, tileserver=args.tileserver)
