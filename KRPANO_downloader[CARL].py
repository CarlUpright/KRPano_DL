import os
import sys
import requests
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QProgressBar, QFileDialog, QInputDialog,
                           QRadioButton, QButtonGroup, QGroupBox, QComboBox, QMessageBox,
                           QSpinBox, QCheckBox, QTabWidget, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont


class DownloadWorker(QThread):
    progress_update = pyqtSignal(int, int)
    status_update = pyqtSignal(str)
    download_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, xml_url, output_dir):
        super().__init__()
        self.xml_url = xml_url
        self.output_dir = output_dir
        self.panoramas = {}
        
    def run(self):
        try:
            self.status_update.emit(f"Parsing XML: {self.xml_url}")
            
            # For Louvre and similar sites, check if we need to fix the base URL
            base_url = self.xml_url[:self.xml_url.rfind('/')+1]
            site_url = self.xml_url
            if not site_url.endswith('/'):
                site_url += '/'
            
            # Try to parse with the normal base URL
            self.panoramas = self.parse_krpano_xml(self.xml_url)
            
            # If no panoramas found, try some common alternate base URLs
            if not self.panoramas:
                alternate_bases = [
                    site_url,  # Try the original site URL
                    base_url,  # Try the XML's directory
                    base_url + "../",  # Try one directory up
                    "https://www.hdmedia.fr/photos360/netvisite/360/",  # For Louvre-like sites
                ]
                
                for alt_base in alternate_bases:
                    self.status_update.emit(f"Trying alternate base URL: {alt_base}")
                    test_panoramas = self.parse_krpano_xml(self.xml_url, alt_base)
                    if test_panoramas:
                        self.panoramas = test_panoramas
                        break
            
            if not self.panoramas:
                self.error_occurred.emit("No panoramas found in the XML file.")
                return
                
            self.status_update.emit(f"Found {len(self.panoramas)} panoramas")
            
            for scene_name, pano_info in self.panoramas.items():
                self.status_update.emit(f"Processing panorama: {scene_name}")
                scene_dir = self.download_panorama({scene_name: pano_info}, self.output_dir)
            
            self.download_complete.emit(self.panoramas)
            
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
    
    def parse_krpano_xml(self, xml_url, override_base_url=None):
        """Parse the KRPano XML to extract panorama info with support for both face and cube formats"""
        response = requests.get(xml_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'xml')
        
        panoramas = {}
        
        # Find all scene elements
        scenes = soup.find_all('scene')
        for scene in scenes:
            scene_name = scene.get('name')
            # Prioritize 'title' attribute for scene name
            scene_title = scene.get('title', '')
            if not scene_title:
                scene_title = scene.get('titleid', scene_name)
            
            # Find the image element within the scene
            image_elem = scene.find('image')
            
            if not image_elem:
                continue
                    
            # Extract tile size and image dimensions
            tile_size = int(image_elem.get('tilesize', 512))
            
            # Check if this is a cube format or the individual faces format
            is_cube_format = image_elem.find('cube') is not None
            is_multires = image_elem.get('multires') == 'true' or image_elem.get('multires') == True
            
            # Handle different ways tiles might be specified
            if is_multires:
                # Find all levels
                levels = image_elem.find_all('level')
                
                if not levels:
                    levels = []
                    for child in image_elem.children:
                        if child.name == 'level':
                            levels.append(child)
                
                if not levels:
                    continue
                    
                # Sort levels by resolution (highest first)
                levels_sorted = sorted(levels, 
                                    key=lambda x: int(x.get('tiledimagewidth', 0)), 
                                    reverse=True)
                
                level = levels_sorted[0]  # Select highest resolution level
                
                tile_width = int(level.get('tiledimagewidth', 0))
                tile_height = int(level.get('tiledimageheight', 0))
                
                # Extract the URL patterns
                if is_cube_format:
                    # Handle cube format with %s, %v, %h placeholders
                    cube_elem = level.find('cube')
                    if not cube_elem:
                        continue
                        
                    cube_url = cube_elem.get('url', '')
                    
                    # Create a face map for the cube format
                    # %s is typically replaced with f, r, b, l, u, d for the 6 faces
                    faces = {
                        'front': cube_url.replace('%s', 'f'),
                        'right': cube_url.replace('%s', 'r'),
                        'back': cube_url.replace('%s', 'b'),
                        'left': cube_url.replace('%s', 'l'),
                        'up': cube_url.replace('%s', 'u'),
                        'down': cube_url.replace('%s', 'd')
                    }
                else:
                    # Handle individual face elements
                    faces = {}
                    for face_name in ['front', 'right', 'back', 'left', 'up', 'down']:
                        face_elem = level.find(face_name)
                        if face_elem and face_elem.get('url'):
                            faces[face_name] = face_elem.get('url')
            else:
                # Handle single-resolution
                tile_width = int(image_elem.get('width', 0))
                tile_height = int(image_elem.get('height', 0))
                
                # If dimensions not found, use a default size
                if tile_width == 0 or tile_height == 0:
                    tile_width = tile_height = 2048  # Common default size
                
                faces = {}
                
                if is_cube_format:
                    # Handle single-resolution cube format
                    cube_elem = image_elem.find('cube')
                    if cube_elem and cube_elem.get('url'):
                        cube_url = cube_elem.get('url')
                        faces = {
                            'front': cube_url.replace('%s', 'f'),
                            'right': cube_url.replace('%s', 'r'),
                            'back': cube_url.replace('%s', 'b'),
                            'left': cube_url.replace('%s', 'l'),
                            'up': cube_url.replace('%s', 'u'),
                            'down': cube_url.replace('%s', 'd')
                        }
                else:
                    # Handle individual face elements
                    for face_name in ['front', 'right', 'back', 'left', 'up', 'down']:
                        face_elem = image_elem.find(face_name)
                        if face_elem and face_elem.get('url'):
                            faces[face_name] = face_elem.get('url')
                
                # Set tile_size to the face size for single-resolution
                tile_size = tile_width if tile_width > 0 else 512
            
            # Calculate number of tiles in each dimension
            # Check if this is a single-image face (no %v/%u or %h in URL)
            is_single_image = all(('%v' not in url and '%u' not in url and '%h' not in url) for url in faces.values())
            
            if is_single_image:
                num_tiles_x = num_tiles_y = 1
            else:
                # For the cube format, we need to estimate based on tile size
                num_tiles_x = (tile_width + tile_size - 1) // tile_size
                num_tiles_y = (tile_height + tile_size - 1) // tile_size
            
            # Use the scene title for both display and folder name
            display_name = scene_title if scene_title else scene_name
            
            if faces:
                # Use override_base_url if provided, otherwise extract from XML URL
                base_url = override_base_url if override_base_url else xml_url[:xml_url.rfind('/')+1]
                
                # For cube format from Louvre, handle the %SWFPATH% differently
                if is_cube_format and "%SWFPATH%" in next(iter(faces.values())):
                    # For Louvre format, extract the base domain and then append the path directly
                    domain_part = xml_url.split('//')[0] + '//' + xml_url.split('//')[1].split('/')[0]
                    path_part = '/'.join(xml_url.split('/')[3:])
                    
                    # Remove 'xml_2015' and anything after it from the path
                    # For Louvre, the correct path pattern is typically:
                    # https://www.louvre.fr/visites-en-ligne/petitegalerie/saison1
                    if 'xml_' in path_part:
                        path_part = path_part.split('xml_')[0]
                    elif 'xml/' in path_part:
                        path_part = path_part.split('xml/')[0]
                    
                    # Remove trailing / if present
                    path_part = path_part.rstrip('/')
                    
                    # Construct the SWFPATH
                    swf_path = f"{domain_part}/{path_part}"
                    
                    # For Louvre formats specifically, we'll use actual preview URL to verify
                    # the correct path
                    test_url = f"{swf_path}/panos/{scene_title}.tiles/preview.jpg"
                    try:
                        # Try to access the preview URL to test our path construction
                        test_response = requests.head(test_url, timeout=3)
                        if test_response.status_code != 200:
                            # If that fails, try without the scene_title and just "01"
                            test_url = f"{swf_path}/panos/01.tiles/preview.jpg"
                            test_response = requests.head(test_url, timeout=3)
                            
                        if test_response.status_code == 200:
                            self.status_update.emit(f"Verified SWFPATH: {swf_path}")
                        else:
                            self.status_update.emit(f"Could not verify path with preview image. Will try anyway.")
                    except Exception as e:
                        self.status_update.emit(f"Error checking preview: {str(e)}. Will try anyway.")
                    
                    # Store the panorama info with correct replacements for SWFPATH
                    panoramas[display_name] = {
                        'title': scene_title,
                        'tile_size': tile_size,
                        'width': tile_width,
                        'height': tile_height,
                        'num_tiles_x': num_tiles_x,
                        'num_tiles_y': num_tiles_y,
                        'faces': {k: v.replace('%SWFPATH%', swf_path) for k, v in faces.items()},
                        'base_url': base_url,
                        'resolution': f"{tile_width}x{tile_height}",
                        'is_single_image': is_single_image,
                        'is_cube_format': is_cube_format,
                        'swf_path': swf_path  # Store for debugging
                    }
                else:
                    # Store the panorama info with the standard format
                    panoramas[display_name] = {
                        'title': scene_title,
                        'tile_size': tile_size,
                        'width': tile_width,
                        'height': tile_height,
                        'num_tiles_x': num_tiles_x,
                        'num_tiles_y': num_tiles_y,
                        'faces': faces,
                        'base_url': base_url,
                        'resolution': f"{tile_width}x{tile_height}",
                        'is_single_image': is_single_image,
                        'is_cube_format': is_cube_format
                    }
                
                # For debugging
                self.status_update.emit(f"Parsed panorama: {display_name}, Tiles: {num_tiles_x}x{num_tiles_y}")
        
        return panoramas
    
    def download_tile(self, url, save_path):
        """Download a single tile image"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return True
        except Exception as e:
            self.status_update.emit(f"Error downloading {url}: {e}")
            return False
    
    def download_panorama(self, panorama_info, output_dir, max_workers=10):
        """Download all tiles for a panorama with support for both face and cube formats"""
        scene_name = list(panorama_info.keys())[0]
        pano = panorama_info[scene_name]
        
        # Create output directory using the scene name
        scene_dir = os.path.join(output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        
        # Track download tasks
        download_tasks = []
        
        # For each face
        face_map = {'front': 2, 'right': 1, 'back': 0, 'left': 3, 'up': 4, 'down': 5}
        cube_face_map = {'f': 'front', 'r': 'right', 'b': 'back', 'l': 'left', 'u': 'up', 'd': 'down'}
        
        for face_name, url_pattern in pano['faces'].items():
            face_num = face_map.get(face_name, 0)
            face_dir = os.path.join(scene_dir, str(face_num))
            os.makedirs(face_dir, exist_ok=True)
            
            # Check if this is a single image face or tiled
            is_single_image = pano.get('is_single_image', False)
            if not is_single_image:
                # Double-check if there are any tile indicators in the URL
                is_single_image = all(marker not in url_pattern for marker in ['%v', '%u', '%h'])
            
            if is_single_image:
                # Single image face - just download the whole face
                tile_url = url_pattern
                if not tile_url.startswith('http'):
                    tile_url = pano['base_url'] + tile_url
                
                # Define save path
                save_path = os.path.join(face_dir, f"0_0.jpg")
                
                # Add to download tasks
                download_tasks.append((tile_url, save_path))
            else:
                # Tiled face - determine the pattern type
                is_cube_format = pano.get('is_cube_format', False)
                
                # Get the correct face identifier for the cube format
                face_id = None
                if is_cube_format:
                    for cube_id, name in cube_face_map.items():
                        if name == face_name and f'%s/{cube_id}/' in url_pattern or f'_{cube_id}_' in url_pattern:
                            face_id = cube_id
                            break
                    
                    # If we couldn't determine the face ID, try to extract it from the URL pattern
                    if not face_id:
                        for cube_id in cube_face_map.keys():
                            if cube_id in url_pattern:
                                face_id = cube_id
                                break
                
                # Download each tile
                for v in range(pano['num_tiles_y']):
                    for u in range(pano['num_tiles_x']):
                        # Create the URL by replacing placeholders
                        # For Louvre format, handle special case of %s/%v template
                        tile_url = url_pattern
                        
                        # Handle each placeholder type
                        if '%s' in tile_url and face_id:
                            tile_url = tile_url.replace('%s', face_id)
                        
                        if '%v' in tile_url:
                            tile_url = tile_url.replace('%v', str(v))
                        
                        if '%h' in tile_url:  # Horizontal index for cube format
                            tile_url = tile_url.replace('%h', str(u))
                        elif '%u' in tile_url:  # Standard format
                            tile_url = tile_url.replace('%u', str(u))
                        
                        # Ensure the URL is absolute
                        if not tile_url.startswith('http'):
                            tile_url = pano['base_url'] + tile_url
                        
                        # Define save path
                        save_path = os.path.join(face_dir, f"{v}_{u}.jpg")
                        
                        # For debugging, log the first URL we're attempting
                        if v == 0 and u == 0:
                            self.status_update.emit(f"First tile URL: {tile_url}")
                        
                        # Add to download tasks
                        download_tasks.append((tile_url, save_path))
        
        # Download tiles in parallel
        successful_downloads = 0
        failures = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.download_tile, url, path) for url, path in download_tasks]
            
            # Show progress bar via signal
            total_tasks = len(download_tasks)
            completed = 0
            
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if future.result():
                    successful_downloads += 1
                else:
                    failures += 1
                self.progress_update.emit(completed, total_tasks)
        
        if failures > 0:
            self.status_update.emit(f"Downloaded {successful_downloads} tiles for {scene_name} with {failures} failures")
        else:
            self.status_update.emit(f"Downloaded {successful_downloads} tiles for {scene_name}")
        
        return scene_dir

class StitchWorker(QThread):
    progress_update = pyqtSignal(int, int)
    status_update = pyqtSignal(str)
    stitch_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, input_dir, scene_name, pano_info=None):
        super().__init__()
        self.input_dir = input_dir
        self.scene_name = scene_name
        self.pano_info = pano_info
        self.face_names = ['front', 'right', 'back', 'left', 'up', 'down']
        self.face_images = []
        
    def run(self):
        try:
            self.status_update.emit(f"Processing panorama: {self.scene_name}")
            scene_dir = os.path.join(self.input_dir, self.scene_name)
            
            if not os.path.exists(scene_dir):
                self.error_occurred.emit(f"Directory not found: {scene_dir}")
                return
            
            # If pano_info wasn't provided, try to infer it
            if not self.pano_info:
                try:
                    # Try to infer width and height from one of the face directories
                    face_dirs = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d)) and d.isdigit()]
                    if not face_dirs:
                        self.error_occurred.emit(f"No face directories found in {scene_dir}")
                        return
                    
                    face_dir = os.path.join(scene_dir, face_dirs[0])
                    tiles = [f for f in os.listdir(face_dir) if f.endswith('.jpg')]
                    if not tiles:
                        self.error_occurred.emit(f"No tiles found in {face_dir}")
                        return
                    
                    # Get the dimensions of one tile
                    sample_tile = Image.open(os.path.join(face_dir, tiles[0]))
                    tile_size = sample_tile.width  # Assuming square tiles
                    
                    # Find the maximum u and v values to determine num_tiles_x and num_tiles_y
                    max_u = max_v = 0
                    for tile in tiles:
                        v, u = map(int, os.path.splitext(tile)[0].split('_'))
                        max_u = max(max_u, u)
                        max_v = max(max_v, v)
                    
                    num_tiles_x = max_u + 1
                    num_tiles_y = max_v + 1
                    
                    # Calculate width and height
                    width = num_tiles_x * tile_size
                    height = num_tiles_y * tile_size
                    
                    self.pano_info = {
                        'tile_size': tile_size,
                        'width': width,
                        'height': height,
                        'num_tiles_x': num_tiles_x,
                        'num_tiles_y': num_tiles_y,
                        'faces': {self.face_names[int(d)]: '' for d in face_dirs if d.isdigit() and int(d) < len(self.face_names)}
                    }
                except Exception as e:
                    self.error_occurred.emit(f"Error inferring panorama info: {str(e)}")
                    return
            
            # Stitch each face
            self.face_images = []
            total_faces = len([f for f in range(6) if os.path.exists(os.path.join(scene_dir, str(f)))])
            face_count = 0
            
            for i, face_name in enumerate(self.face_names):
                face_dir = os.path.join(scene_dir, str(i))
                if os.path.exists(face_dir) and os.listdir(face_dir):
                    self.status_update.emit(f"Stitching {face_name} face...")
                    face_img = self.stitch_cubemap_face(face_dir, self.pano_info, i)
                    face_output_path = os.path.join(scene_dir, f"face_{face_name}.jpg")
                    face_img.save(face_output_path)
                    self.face_images.append(face_img)
                    face_count += 1
                    self.progress_update.emit(face_count, total_faces)
                    self.status_update.emit(f"Saved {face_output_path}")
            
            self.stitch_complete.emit(self.face_images)
            
        except Exception as e:
            self.error_occurred.emit(f"Error during stitching: {str(e)}")
    
    def stitch_cubemap_face(self, face_dir, pano_info, face_num):
        """Stitch tiles to create a single face of the cubemap with proper index handling"""
        # Scan the directory for all tiles with pattern v_u.jpg
        tiles = {}
        for file in os.listdir(face_dir):
            if file.endswith('.jpg'):
                try:
                    # Extract v and u coordinates from filename (format: v_u.jpg)
                    v, u = map(int, os.path.splitext(file)[0].split('_'))
                    tiles[(v, u)] = file
                except:
                    continue
        
        # If no tiles found, return empty image
        if not tiles:
            self.status_update.emit(f"No tiles found in {face_dir}")
            return Image.new('RGB', (pano_info['width'], pano_info['height']))
        
        # Determine actual grid size from the filenames
        max_v = max(v for v, u in tiles.keys())
        max_u = max(u for v, u in tiles.keys())
        min_v = min(v for v, u in tiles.keys())
        min_u = min(u for v, u in tiles.keys())
        
        # Load a sample tile to get the tile dimensions
        sample_key = next(iter(tiles))
        sample_tile = Image.open(os.path.join(face_dir, tiles[sample_key]))
        tile_width, tile_height = sample_tile.size
        
        # Calculate the total image dimensions
        total_width = (max_u - min_u + 1) * tile_width
        total_height = (max_v - min_v + 1) * tile_height
        
        self.status_update.emit(f"Creating face {face_num} with grid {min_v}:{max_v} x {min_u}:{max_u} (size: {total_width}x{total_height})")
        
        # Create a blank image
        face_img = Image.new('RGB', (total_width, total_height))
        
        # Place each tile in the correct position
        for (v, u), file in tiles.items():
            try:
                # Calculate position (adjust for min_v and min_u to handle non-zero starting indices)
                x = (u - min_u) * tile_width
                y = (v - min_v) * tile_height
                
                # Open and paste the tile
                tile = Image.open(os.path.join(face_dir, file))
                face_img.paste(tile, (x, y))
            except Exception as e:
                self.status_update.emit(f"Error processing tile {file}: {e}")
        
        # Resize to the expected dimensions if needed
        # This ensures all faces have consistent dimensions
        expected_width, expected_height = pano_info['width'], pano_info['height']
        if total_width != expected_width or total_height != expected_height:
            self.status_update.emit(f"Resizing face from {total_width}x{total_height} to {expected_width}x{expected_height}")
            face_img = face_img.resize((expected_width, expected_height))
        
        return face_img


class EquirectangularWorker(QThread):
    progress_update = pyqtSignal(int, int)
    status_update = pyqtSignal(str)
    conversion_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, cubemap_faces, output_dir, scene_name, output_width=8192, output_height=4096):
        super().__init__()
        self.cubemap_faces = cubemap_faces
        self.output_dir = output_dir
        self.scene_name = scene_name
        self.output_width = output_width
        self.output_height = output_height
        
    def run(self):
        try:
            self.status_update.emit("Converting to equirectangular projection...")
            
            # Get the face size (assuming all faces are the same size)
            face_size = self.cubemap_faces[0].size[0]
            
            # Pre-process the faces with correct rotations and flips
            processed_faces = []
            for i, face in enumerate(self.cubemap_faces):
                face_np = np.array(face)
                
                if i == 0:  # front
                    # No change needed
                    processed_faces.append(face_np)
                elif i == 1:  # right
                    # No change needed
                    processed_faces.append(face_np)
                elif i == 2:  # back
                    # No change needed
                    processed_faces.append(face_np)
                elif i == 3:  # left
                    # No change needed
                    processed_faces.append(face_np)
                elif i == 4:  # up
                    # Rotate up face 90 degrees clockwise
                    processed_faces.append(np.rot90(face_np, k=3))
                elif i == 5:  # down
                    # Rotate down face 90 degrees counter-clockwise
                    processed_faces.append(np.rot90(face_np, k=1))
            
            # Convert numpy arrays to OpenCV format (BGR)
            faces = []
            for face in processed_faces:
                faces.append(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            
            # Create output equirectangular image
            equirectangular = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            
            # For each pixel in the equirectangular image
            for y in range(self.output_height):
                if y % 10 == 0:  # Update progress every 10 rows to avoid too many signals
                    self.progress_update.emit(y, self.output_height)
                    
                for x in range(self.output_width):
                    # Convert to spherical coordinates
                    theta = 2 * np.pi * x / self.output_width - np.pi
                    phi = np.pi * y / self.output_height - np.pi/2
                    
                    # Convert to 3D coordinates
                    x3d = np.cos(phi) * np.cos(theta)
                    y3d = np.cos(phi) * np.sin(theta)
                    z3d = np.sin(phi)
                    
                    # Determine which face to sample from
                    abs_x, abs_y, abs_z = abs(x3d), abs(y3d), abs(z3d)
                    max_val = max(abs_x, abs_y, abs_z)
                    
                    # Front face (positive x)
                    if max_val == abs_x and x3d > 0:
                        face_idx = 0
                        u = (-y3d / abs_x + 1) / 2
                        v = (-z3d / abs_x + 1) / 2
                    # Right face (positive y)
                    elif max_val == abs_y and y3d > 0:
                        face_idx = 1
                        u = (x3d / abs_y + 1) / 2
                        v = (-z3d / abs_y + 1) / 2
                    # Back face (negative x)
                    elif max_val == abs_x and x3d < 0:
                        face_idx = 2
                        u = (y3d / abs_x + 1) / 2
                        v = (-z3d / abs_x + 1) / 2
                    # Left face (negative y)
                    elif max_val == abs_y and y3d < 0:
                        face_idx = 3
                        u = (-x3d / abs_y + 1) / 2
                        v = (-z3d / abs_y + 1) / 2
                    # Up face (positive z)
                    elif max_val == abs_z and z3d > 0:
                        face_idx = 4
                        u = (x3d / abs_z + 1) / 2
                        v = (y3d / abs_z + 1) / 2
                    # Down face (negative z)
                    else:
                        face_idx = 5
                        u = (x3d / abs_z + 1) / 2
                        v = (-y3d / abs_z + 1) / 2
                    
                    # Sample from the appropriate face
                    fx = int(u * face_size)
                    fy = int(v * face_size)
                    
                    # Clamp values
                    fx = max(0, min(face_size - 1, fx))
                    fy = max(0, min(face_size - 1, fy))
                    
                    # Copy pixel
                    equirectangular[y, x] = faces[face_idx][fy, fx]
            
            # Convert back to PIL and rotate
            equirectangular_img = Image.fromarray(cv2.cvtColor(equirectangular, cv2.COLOR_BGR2RGB))
            equirectangular_img = equirectangular_img.rotate(180)
            
            equi_output_path = os.path.join(self.output_dir, f"{self.scene_name}_equirectangular.jpg")
            equirectangular_img.save(equi_output_path)
            
            self.status_update.emit(f"Saved equirectangular panorama to {equi_output_path}")
            self.conversion_complete.emit(equi_output_path)
            
        except Exception as e:
            self.error_occurred.emit(f"Error during equirectangular conversion: {str(e)}")


class KrpanoDownloaderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # State variables
        self.panoramas = {}
        self.current_scene = None
        self.face_images = []
        self.download_worker = None
        self.stitch_worker = None
        self.equi_worker = None
        
    def init_ui(self):
        self.setWindowTitle('KRPANO Downloader')
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Download tab
        download_tab = QWidget()
        download_layout = QVBoxLayout()
        
        # Input method selection (URL or Local Folder)
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout()
        
        self.url_radio = QRadioButton("Download from URL")
        self.local_radio = QRadioButton("Use Local Folder")
        self.url_radio.setChecked(True)
        
        input_layout.addWidget(self.url_radio)
        input_layout.addWidget(self.local_radio)
        input_group.setLayout(input_layout)
        
        # Connect radio buttons to show/hide relevant inputs
        self.url_radio.toggled.connect(self.toggle_input_method)
        
        # URL input
        self.url_group = QGroupBox("URL Input")
        url_layout = QVBoxLayout()
        
        url_input_layout = QHBoxLayout()
        url_input_layout.addWidget(QLabel("XML URL:"))
        self.url_input = QLineEdit()
        url_input_layout.addWidget(self.url_input)
        
        url_layout.addLayout(url_input_layout)
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(os.path.join(os.path.expanduser("~"), "KRPANO_Output"))
        output_dir_layout.addWidget(self.output_dir_input)
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.browse_output_btn)
        
        url_layout.addLayout(output_dir_layout)
        
        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.start_download)
        url_layout.addWidget(self.download_btn)
        
        self.url_group.setLayout(url_layout)
        
        # Local folder input
        self.local_group = QGroupBox("Local Folder Input")
        local_layout = QVBoxLayout()
        
        local_dir_layout = QHBoxLayout()
        local_dir_layout.addWidget(QLabel("Input Directory:"))
        self.local_dir_input = QLineEdit()
        local_dir_layout.addWidget(self.local_dir_input)
        self.browse_local_btn = QPushButton("Browse...")
        self.browse_local_btn.clicked.connect(self.browse_local_dir)
        local_dir_layout.addWidget(self.browse_local_btn)
        
        local_layout.addLayout(local_dir_layout)
        
        self.scenes_combo = QComboBox()
        self.scenes_combo.setEnabled(False)
        local_layout.addWidget(QLabel("Select Panorama:"))
        local_layout.addWidget(self.scenes_combo)
        
        self.load_local_btn = QPushButton("Load Scenes")
        self.load_local_btn.clicked.connect(self.load_local_scenes)
        local_layout.addWidget(self.load_local_btn)
        
        self.local_group.setLayout(local_layout)
        self.local_group.setVisible(False)
        
        # Progress area
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        
        # Add all components to download tab
        download_layout.addWidget(input_group)
        download_layout.addWidget(self.url_group)
        download_layout.addWidget(self.local_group)
        download_layout.addWidget(progress_group)
        download_layout.addStretch()
        
        download_tab.setLayout(download_layout)
        
        # Equirectangular tab
        equi_tab = QWidget()
        equi_layout = QVBoxLayout()
        
        # Panorama selection
        pano_select_layout = QHBoxLayout()
        pano_select_layout.addWidget(QLabel("Panorama:"))
        self.equi_pano_combo = QComboBox()
        self.equi_pano_combo.setEnabled(False)
        pano_select_layout.addWidget(self.equi_pano_combo)
        
        self.refresh_panos_btn = QPushButton("Refresh")
        self.refresh_panos_btn.clicked.connect(self.refresh_panoramas)
        pano_select_layout.addWidget(self.refresh_panos_btn)
        
        equi_layout.addLayout(pano_select_layout)
        
        # Resolution selection
        resolution_group = QGroupBox("Output Resolution")
        resolution_layout = QVBoxLayout()
        
        self.max_res_label = QLabel("Maximum resolution: N/A")
        resolution_layout.addWidget(self.max_res_label)
        
        # Width and height inputs
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.width_input = QSpinBox()
        self.width_input.setRange(1024, 26664)
        self.width_input.setValue(8192)
        self.width_input.setSingleStep(1024)
        width_layout.addWidget(self.width_input)
        
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        self.height_input = QSpinBox()
        self.height_input.setRange(512, 13332)
        self.height_input.setValue(4096)
        self.height_input.setSingleStep(512)
        height_layout.addWidget(self.height_input)
        
        resolution_layout.addLayout(width_layout)
        resolution_layout.addLayout(height_layout)
        
        # Maintain aspect ratio
        self.maintain_aspect = QCheckBox("Maintain aspect ratio (2:1)")
        self.maintain_aspect.setChecked(True)
        self.maintain_aspect.toggled.connect(self.update_aspect_ratio)
        resolution_layout.addWidget(self.maintain_aspect)
        
        # Link width and height when maintaining aspect ratio
        self.width_input.valueChanged.connect(self.width_changed)
        
        resolution_group.setLayout(resolution_layout)
        
        # Convert button
        self.convert_btn = QPushButton("Convert to Equirectangular")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(False)
        
        # Equirectangular progress
        equi_progress_group = QGroupBox("Conversion Progress")
        equi_progress_layout = QVBoxLayout()
        
        self.equi_status_label = QLabel("Ready")
        equi_progress_layout.addWidget(self.equi_status_label)
        
        self.equi_progress_bar = QProgressBar()
        self.equi_progress_bar.setRange(0, 100)
        self.equi_progress_bar.setValue(0)
        equi_progress_layout.addWidget(self.equi_progress_bar)
        
        equi_progress_group.setLayout(equi_progress_layout)
        
        # Add all components to equirectangular tab
        equi_layout.addWidget(resolution_group)
        equi_layout.addWidget(self.convert_btn)
        equi_layout.addWidget(equi_progress_group)
        equi_layout.addStretch()
        
        equi_tab.setLayout(equi_layout)
        
        # Add tabs to tab widget
        self.tabs.addTab(download_tab, "Download")
        self.tabs.addTab(equi_tab, "Convert to Equirectangular")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar for important messages
        self.statusBar().showMessage("Ready")
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def toggle_input_method(self):
        """Toggle between URL and local folder input methods"""
        self.url_group.setVisible(self.url_radio.isChecked())
        self.local_group.setVisible(self.local_radio.isChecked())
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_input.setText(directory)
    
    def browse_local_dir(self):
        """Browse for local input directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.local_dir_input.setText(directory)
    
    def load_local_scenes(self):
        """Load panorama scenes from local directory"""
        input_dir = self.local_dir_input.text()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Invalid Directory", "Please select a valid directory")
            return
        
        # Find subdirectories that might be panorama scenes
        self.scenes_combo.clear()
        scene_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        
        if not scene_dirs:
            QMessageBox.warning(self, "No Scenes Found", "No panorama scenes found in the selected directory")
            return
        
        # Add scenes to combo box
        for scene in scene_dirs:
            face_dirs = [d for d in os.listdir(os.path.join(input_dir, scene)) 
                       if os.path.isdir(os.path.join(input_dir, scene, d)) and d.isdigit()]
            
            if face_dirs:  # Only add if at least one face directory exists
                self.scenes_combo.addItem(scene)
        
        if self.scenes_combo.count() > 0:
            self.scenes_combo.setEnabled(True)
            self.refresh_panoramas()
            self.status_label.setText(f"Found {self.scenes_combo.count()} panorama scenes")
        else:
            QMessageBox.warning(self, "No Scenes Found", "No valid panorama scenes found in the selected directory")
    
    def refresh_panoramas(self):
        """Refresh the list of panoramas for equirectangular conversion"""
        self.equi_pano_combo.clear()
        
        if self.url_radio.isChecked():
            # Use downloaded panoramas
            for scene_name in self.panoramas.keys():
                self.equi_pano_combo.addItem(scene_name)
        else:
            # Use local panoramas
            input_dir = self.local_dir_input.text()
            if not input_dir or not os.path.isdir(input_dir):
                return
            
            for i in range(self.scenes_combo.count()):
                scene = self.scenes_combo.itemText(i)
                self.equi_pano_combo.addItem(scene)
        
        if self.equi_pano_combo.count() > 0:
            self.equi_pano_combo.setEnabled(True)
            self.convert_btn.setEnabled(True)
            self.update_max_resolution()
        else:
            self.equi_pano_combo.setEnabled(False)
            self.convert_btn.setEnabled(False)
            self.max_res_label.setText("Maximum resolution: N/A")
    
    def update_max_resolution(self):
        """Update the maximum resolution label based on the selected panorama"""
        if self.equi_pano_combo.count() == 0:
            return
        
        scene_name = self.equi_pano_combo.currentText()
        
        if self.url_radio.isChecked() and scene_name in self.panoramas:
            # Get resolution from downloaded panorama info
            pano_info = self.panoramas[scene_name]
            face_size = pano_info['width']  # Assuming square faces
            max_width = face_size * 4  # 4 times the face width is a good equirectangular width
            max_height = face_size * 2  # 2 times the face height for 2:1 aspect ratio
            
            self.max_res_label.setText(f"Maximum recommended resolution: {max_width}x{max_height}")
            
            # Update spin boxes with maximum values
            if max_width > self.width_input.value():
                self.width_input.setValue(max_width)
            if max_height > self.height_input.value():
                self.height_input.setValue(max_height)
            
        else:
            # Try to infer from local files
            input_dir = self.local_dir_input.text()
            scene_dir = os.path.join(input_dir, scene_name)
            
            if not os.path.isdir(scene_dir):
                return
            
            # Try to find a face image
            face_files = [f for f in os.listdir(scene_dir) if f.startswith("face_") and f.endswith(".jpg")]
            
            if face_files:
                # Get the size of the first face
                face_img = Image.open(os.path.join(scene_dir, face_files[0]))
                face_width, face_height = face_img.size
                
                max_width = face_width * 4
                max_height = face_height * 2
                
                self.max_res_label.setText(f"Maximum recommended resolution: {max_width}x{max_height}")
                
                # Update spin boxes with maximum values
                if max_width > self.width_input.value():
                    self.width_input.setValue(max_width)
                if max_height > self.height_input.value():
                    self.height_input.setValue(max_height)
            else:
                # Try to find face directories and infer size
                face_dirs = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d)) and d.isdigit()]
                
                if not face_dirs:
                    self.max_res_label.setText("Maximum resolution: Unknown")
                    return
                
                face_dir = os.path.join(scene_dir, face_dirs[0])
                tiles = [f for f in os.listdir(face_dir) if f.endswith('.jpg')]
                
                if not tiles:
                    self.max_res_label.setText("Maximum resolution: Unknown")
                    return
                
                # Get the dimensions of one tile
                sample_tile = Image.open(os.path.join(face_dir, tiles[0]))
                tile_size = sample_tile.width  # Assuming square tiles
                
                # Find the maximum u and v values
                max_u = max_v = 0
                for tile in tiles:
                    v, u = map(int, os.path.splitext(tile)[0].split('_'))
                    max_u = max(max_u, u)
                    max_v = max(max_v, v)
                
                # Calculate face size
                face_width = (max_u + 1) * tile_size
                face_height = (max_v + 1) * tile_size
                
                # Calculate equirectangular size
                max_width = face_width * 4
                max_height = face_height * 2
                
                self.max_res_label.setText(f"Maximum recommended resolution: {max_width}x{max_height}")
                
                # Update spin boxes with maximum values
                if max_width > self.width_input.value():
                    self.width_input.setValue(max_width)
                if max_height > self.height_input.value():
                    self.height_input.setValue(max_height)
    
    def width_changed(self, value):
        """Update height when width changes (to maintain aspect ratio)"""
        if self.maintain_aspect.isChecked():
            self.height_input.setValue(value // 2)
    
    def update_aspect_ratio(self, checked):
        """Update height when aspect ratio checkbox changes"""
        if checked:
            self.height_input.setValue(self.width_input.value() // 2)
    
    def start_download(self):
        """Start the download process"""
        xml_url = self.url_input.text()
        output_dir = self.output_dir_input.text()
        
        if not xml_url:
            QMessageBox.warning(self, "Invalid URL", "Please enter the direct URL to the XML file")
            return
        
        if not output_dir:
            QMessageBox.warning(self, "Invalid Directory", "Please select a valid output directory")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting download...")
        
        # Create and start worker thread
        self.download_worker = DownloadWorker(xml_url, output_dir)
        self.download_worker.progress_update.connect(self.update_download_progress)
        self.download_worker.status_update.connect(self.update_status)
        self.download_worker.download_complete.connect(self.download_completed)
        self.download_worker.error_occurred.connect(self.show_error)
        
        self.download_worker.start()
        
        # Disable download button while processing
        self.download_btn.setEnabled(False)
    
    def update_download_progress(self, current, total):
        """Update the progress bar for download progress"""
        progress = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
    
    def update_status(self, message):
        """Update the status label"""
        self.status_label.setText(message)
        self.statusBar().showMessage(message, 3000)
    
    def download_completed(self, panoramas):
        """Handle download completion"""
        self.panoramas = panoramas
        self.download_btn.setEnabled(True)
        self.status_label.setText("Download completed")
        self.progress_bar.setValue(100)
        
        # Update the equirectangular tab
        self.refresh_panoramas()
        
        # Switch to equirectangular tab
        self.tabs.setCurrentIndex(1)
    
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.download_btn.setEnabled(True)
        self.status_label.setText("Error occurred")
    
    def start_conversion(self):
        """Start the equirectangular conversion process"""
        if self.equi_pano_combo.count() == 0:
            return
        
        scene_name = self.equi_pano_combo.currentText()
        output_width = self.width_input.value()
        output_height = self.height_input.value()
        
        if self.url_radio.isChecked():
            # Use downloaded panoramas
            input_dir = self.output_dir_input.text()
            pano_info = self.panoramas.get(scene_name)
            
            # Use folder_name for file operations if available
            if pano_info and 'folder_name' in pano_info:
                scene_dir_name = pano_info['folder_name']
            else:
                scene_dir_name = scene_name
        else:
            # Use local panoramas
            input_dir = self.local_dir_input.text()
            pano_info = None
            scene_dir_name = scene_name
        
        # Reset progress
        self.equi_progress_bar.setValue(0)
        self.equi_status_label.setText("Starting stitching...")
        
        # First stitch the faces if needed
        self.stitch_worker = StitchWorker(input_dir, scene_dir_name, pano_info)
        self.stitch_worker.progress_update.connect(self.update_stitch_progress)
        self.stitch_worker.status_update.connect(self.update_equi_status)
        self.stitch_worker.stitch_complete.connect(lambda faces: self.stitch_completed(faces, input_dir, scene_name, output_width, output_height))
        self.stitch_worker.error_occurred.connect(self.show_equi_error)
        
        self.stitch_worker.start()
        
        # Disable convert button while processing
        self.convert_btn.setEnabled(False)
    
    def update_stitch_progress(self, current, total):
        """Update the progress bar for stitching progress"""
        progress = int(current / total * 50) if total > 0 else 0  # Use first half of progress bar for stitching
        self.equi_progress_bar.setValue(progress)
    
    def update_equi_status(self, message):
        """Update the equirectangular status label"""
        self.equi_status_label.setText(message)
        self.statusBar().showMessage(message, 3000)
    
    def stitch_completed(self, face_images, input_dir, scene_name, output_width, output_height):
        """Handle stitching completion and start equirectangular conversion"""
        self.face_images = face_images
        
        if len(face_images) < 6:
            self.show_equi_error(f"Not enough faces found (need 6, found {len(face_images)})")
            self.convert_btn.setEnabled(True)
            return
        
        self.equi_status_label.setText("Stitching completed, starting equirectangular conversion...")
        
        # Now start the equirectangular conversion
        if self.url_radio.isChecked():
            output_dir = self.output_dir_input.text()
        else:
            output_dir = self.local_dir_input.text()
        
        self.equi_worker = EquirectangularWorker(face_images, output_dir, scene_name, output_width, output_height)
        self.equi_worker.progress_update.connect(self.update_equi_progress)
        self.equi_worker.status_update.connect(self.update_equi_status)
        self.equi_worker.conversion_complete.connect(self.conversion_completed)
        self.equi_worker.error_occurred.connect(self.show_equi_error)
        
        self.equi_worker.start()
    
    def update_equi_progress(self, current, total):
        """Update the progress bar for equirectangular conversion progress"""
        # Use second half of progress bar for conversion (50-100%)
        progress = 50 + int(current / total * 50) if total > 0 else 50
        self.equi_progress_bar.setValue(progress)
    
    def conversion_completed(self, output_path):
        """Handle equirectangular conversion completion"""
        self.convert_btn.setEnabled(True)
        self.equi_status_label.setText("Conversion completed")
        self.equi_progress_bar.setValue(100)
        
        QMessageBox.information(self, "Conversion Complete", 
                               f"Equirectangular panorama saved to:\n{output_path}")
    
    def show_equi_error(self, message):
        """Show equirectangular conversion error message"""
        QMessageBox.critical(self, "Error", message)
        self.convert_btn.setEnabled(True)
        self.equi_status_label.setText("Error occurred")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    # Set app font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = KrpanoDownloaderGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
