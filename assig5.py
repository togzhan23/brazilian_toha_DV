
import os
import sys
import traceback
import numpy as np
import copy

MODEL_PATH = "12140_Skull_v3_L2.obj" 

TEMP_CONVERTED_PATH = "converted_for_open3d.ply"
TEMP_PCD_PATH = "temp_from_mesh.ply"
VOXEL_SIZE = 0.05
POISSON_DEPTH = 9
GRADIENT_AXIS = "z" 
PLANE_OFFSET = 0.0
SKIP_VISUAL = False  


def safe_draw(geometries, window_name="Open3D"):
    """Попытаться открыть окно визуализации, но не падать при ошибке (macOS headless issues)."""
    if SKIP_VISUAL:
        print("[INFO] Визуализация пропущена (SKIP_VISUAL=True).")
        return
    try:
        import open3d as o3d
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
    except Exception as e:
        print(f"[WARNING] Визуализация не удалась ({e}). Продолжаю без окна.")

def print_header(title):
    print("\n" + "="*40)
    print(title)
    print("="*40)

def mesh_info(mesh):
    import open3d as o3d
    v = np.asarray(mesh.vertices).shape[0]
    t = np.asarray(mesh.triangles).shape[0]
    return v, t, mesh.has_vertex_colors(), mesh.has_vertex_normals()

def pcd_info(pcd):
    v = np.asarray(pcd.points).shape[0]
    return v, pcd.has_colors(), pcd.has_normals()

def try_trimesh_convert(in_path, out_path):
  
    try:
        import trimesh
    except Exception as e:
        print("[INFO] trimesh не установлен. Установите: python3 -m pip install --user trimesh shapely")
        return False

    try:
        print(f"[INFO] Пробуем загрузить {in_path} через trimesh...")
        scene = trimesh.load(in_path, force='scene')
        if isinstance(scene, trimesh.Scene):
            geoms = list(scene.geometry.values())
            if len(geoms) == 0:
                print("[ERROR] trimesh загрузил сцену, но геометрий нет.")
                return False
            # объединяем все геометрии в один mesh
            try:
                mesh = trimesh.util.concatenate(geoms)
                print(f"[INFO] Объединено {len(geoms)} геометрий в один mesh.")
            except Exception as e:
                print("[WARNING] Не удалось объединить геометрии напрямую:", e)
                mesh = geoms[0]
        else:
            mesh = scene

        if getattr(mesh, 'is_empty', False):
            print("[ERROR] trimesh: mesh пустой после загрузки.")
            return False

        # Попытка триангуляции (best-effort)
        try:
            mesh = mesh.triangulate()
            print("[INFO] trimesh.triangulate() выполнена.")
        except Exception as e:
            print("[WARNING] trimesh.triangulate() не сработала:", e)

        # Экспорт в ply
        mesh.export(out_path)
        print(f"[INFO] Успешно экспортирован {out_path}")
        return True

    except Exception as e:
        print("[ERROR] Ошибка при работе с trimesh:")
        traceback.print_exc()
        return False

def color_mesh_from_pcd(mesh, pcd):
    
    import open3d as o3d
    if not pcd.has_colors():
        print("[color_mesh_from_pcd] pcd не содержит цветов. Пропускаем перенос.")
        return mesh
    pcd_pts = np.asarray(pcd.points)
    pcd_cols = np.asarray(pcd.colors)
    tree = o3d.geometry.KDTreeFlann(pcd)
    verts = np.asarray(mesh.vertices)
    colors = np.zeros((len(verts), 3))
    # nearest neighbor for each vertex
    for i, v in enumerate(verts):
        [k, idx, _] = tree.search_knn_vector_3d(v, 1)
        if k > 0:
            colors[i] = pcd_cols[idx[0]]
        else:
            colors[i] = [0.5, 0.5, 0.5]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def main():
    # проверка файла
    print_header("Проверка входного файла")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Файл не найден: {MODEL_PATH}")
        print("Укажите MODEL_PATH абсолютным путем к вашей модели.")
        sys.exit(1)
    print("Файл найден:", MODEL_PATH)
    print("Размер (байт):", os.path.getsize(MODEL_PATH))

    # Импорт Open3D
    try:
        import open3d as o3d
        print("Open3D version:", o3d.__version__)
    except Exception as e:
        print("[ERROR] Не удалось импортировать open3d. Установите open3d: python3 -m pip install open3d")
        traceback.print_exc()
        sys.exit(1)

    # Попытка прочитать mesh
    print_header("Шаг 1: Чтение mesh")
    mesh = None
    try:
        mesh = o3d.io.read_triangle_mesh(MODEL_PATH)
        if mesh is None:
            print("[WARN] read_triangle_mesh вернул None.")
            mesh = o3d.geometry.TriangleMesh()
    except Exception as e:
        print("[WARN] read_triangle_mesh бросил исключение:", e)
        mesh = o3d.geometry.TriangleMesh()

    v, t = len(mesh.vertices), len(mesh.triangles)
    print(f"Первичная информация: vertices={v}, triangles={t}")

    # Если mesh пустой — попробуем прочитать как point cloud
    pcd = None
    if v == 0 or t == 0:
        print("[INFO] Mesh пустой или без треугольников. Пробуем read_point_cloud()...")
        try:
            pcd = o3d.io.read_point_cloud(MODEL_PATH)
            if pcd is None:
                pcd = o3d.geometry.PointCloud()
        except Exception as e:
            print("[WARN] read_point_cloud бросил исключение:", e)
            pcd = o3d.geometry.PointCloud()

        vp = len(pcd.points)
        print(f"Point cloud: points={vp}")
        if vp == 0:
            print("[INFO] point cloud пустой. Попытка автоконвертации через trimesh...")
            converted_ok = try_trimesh_convert(MODEL_PATH, TEMP_CONVERTED_PATH)
            if converted_ok:
                print("[INFO] Перезагружаем mesh из сконвертированного PLY...")
                mesh = o3d.io.read_triangle_mesh(TEMP_CONVERTED_PATH)
                v, t = len(mesh.vertices), len(mesh.triangles)
                print(f"После конвертации: vertices={v}, triangles={t}")
                if v == 0 or t == 0:
                    print("[ERROR] Даже после конвертации mesh пустой.")
                    print("Рекомендация: попробуйте открыть файл в Blender и экспортировать PLY с триангуляцией.")
                else:
                    pass
            else:
                print("[ERROR] Автоконвертация не удалась. Пожалуйста, используйте Blender или скачайте другой файл.")
        else:
            pass
    else:
        pass

    # Если mesh есть и не пуст — визуализируем и продолжаем
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        try:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
        except Exception:
            pass
        print_header("Шаг 1: Визуализация исходного mesh")
        print("Количество вершин:", len(mesh.vertices))
        print("Количество треугольников:", len(mesh.triangles))
        print("Has colors:", mesh.has_vertex_colors())
        print("Has normals:", mesh.has_vertex_normals())
        safe_draw([mesh], window_name="Step 1: Original Mesh")
    else:
        # если mesh пуст, но есть pcd, работаем с pcd
        if pcd is not None and len(pcd.points) > 0:
            print_header("Шаг 1b: У нас есть point cloud, но не mesh")
            print("Points:", len(pcd.points))
            safe_draw([pcd], window_name="Step 1b: Point Cloud (input)")
        else:
            print("[ERROR] Нет корректной геометрии (ни mesh, ни point cloud). Завершаю.")
            sys.exit(1)

    #  ШАГ 2 
    print_header("Шаг 2: Преобразование в облако точек (если исходный mesh)")
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        try:
            SAMPLE_N = 200000
            SAMPLE_N = min(SAMPLE_N, max(1000, int(len(mesh.triangles)*2)))  # разумный cap
            pcd_from_mesh = mesh.sample_points_uniformly(number_of_points=SAMPLE_N)
            o3d.io.write_point_cloud(TEMP_PCD_PATH, pcd_from_mesh)
            pcd = o3d.io.read_point_cloud(TEMP_PCD_PATH)
            print("Сэмплированные точки:", len(pcd.points))
            safe_draw([pcd], window_name="Step 2: Point Cloud (from mesh)")
        except Exception as e:
            print("[ERROR] Не удалось сэмплировать точки из mesh:", e)
            traceback.print_exc()
    else:
        if pcd is None or len(pcd.points) == 0:
            print("[ERROR] Нет point cloud для дальнейшей обработки.")
            sys.exit(1)

    # ШАГ 3
    print_header("Шаг 3: Реконструкция поверхности (Poisson) и перенос цвета")
    mesh_poisson_cropped = None
    try:
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=POISSON_DEPTH)
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.02, bbox.get_center())
        mesh_poisson_cropped = mesh_poisson.crop(bbox)

        if not mesh_poisson_cropped.has_vertex_normals():
            mesh_poisson_cropped.compute_vertex_normals()

        if pcd.has_colors():
            try:
                mesh_poisson_cropped = color_mesh_from_pcd(mesh_poisson_cropped, pcd)
                print("[INFO] Цвета перенесены из pcd в Poisson mesh.")
            except Exception as e:
                print("[WARN] Перенос цвета не удался:", e)
        else:
            verts = np.asarray(mesh_poisson_cropped.vertices)
            axis_idx = {"x":0,"y":1,"z":2}.get(GRADIENT_AXIS.lower(), 2)
            vals = verts[:, axis_idx]
            if vals.ptp() == 0:
                norm_vals = np.zeros_like(vals)
            else:
                norm_vals = (vals - vals.min()) / vals.ptp()
            cols = np.zeros((len(vals),3))
            cols[:,0] = norm_vals
            cols[:,1] = 1.0 - norm_vals
            cols[:,2] = 0.5
            mesh_poisson_cropped.vertex_colors = o3d.utility.Vector3dVector(cols)
            print("[INFO] Fallback: назначен градиентный цвет для Poisson mesh.")

        print("Poisson mesh (cropped): vertices=", len(mesh_poisson_cropped.vertices), "triangles=", len(mesh_poisson_cropped.triangles))
        safe_draw([mesh_poisson_cropped], window_name="Step 3: Poisson Reconstructed (colored & cropped)")
    except Exception as e:
        print("[ERROR] Poisson reconstruction failed:", e)
        traceback.print_exc()

    # ШАГ 4
    print_header("Шаг 4: Вокселизация (VoxelGrid)")
    try:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
        print("Voxels count:", len(voxel_grid.get_voxels()))
        safe_draw([voxel_grid], window_name="Step 4: Voxel Grid")
    except Exception as e:
        print("[ERROR] Voxelization failed:", e)
        traceback.print_exc()

    # ШАГ 5
    print_header("Шаг 5: Добавление плоскости (cutting plane)")
    try:
        if mesh_poisson_cropped is not None and len(mesh_poisson_cropped.vertices) > 0:
            ref_bbox = mesh_poisson_cropped.get_axis_aligned_bounding_box()
        elif pcd is not None and len(pcd.points) > 0:
            ref_bbox = pcd.get_axis_aligned_bounding_box()
        else:
            ref_bbox = mesh.get_axis_aligned_bounding_box()

        cent = ref_bbox.get_center()
        extent = ref_bbox.get_extent()

        plane_w = extent[2] * 2.5 
        plane_h = extent[1] * 2.5  
        plane_th = max(0.001, extent[0] * 0.01)  

        plane = o3d.geometry.TriangleMesh.create_box(width=plane_th, height=plane_h, depth=plane_w)
        plane.compute_vertex_normals()
        plane.translate(cent - plane.get_center())
        plane.paint_uniform_color([0.2, 0.2, 0.2])

        if mesh_poisson_cropped is not None and len(mesh_poisson_cropped.vertices) > 0:
            safe_draw([mesh_poisson_cropped, plane], window_name="Step 5: Mesh + Cutting Plane")
        elif len(mesh.vertices) > 0:
            safe_draw([mesh, plane], window_name="Step 5: Mesh + Plane")
        else:
            safe_draw([pcd, plane], window_name="Step 5: PCD + Plane")

        print("Plane center:", plane.get_center())
    except Exception as e:
        print("[ERROR] Не удалось создать/разместить плоскость:", e)
        traceback.print_exc()

    # ШАГ 6 
    print_header("Шаг 6: Обрезка по плоскости (удаляем правую половину)")
    try:
        plane_x = float(np.asarray(plane.get_center())[0])
        print("Использую plane_x =", plane_x)

        if mesh_poisson_cropped is not None and len(mesh_poisson_cropped.vertices) > 0:
            src_mesh = mesh_poisson_cropped
        elif len(mesh.vertices) > 0:
            src_mesh = mesh
        else:
            if pcd is not None and len(pcd.points) > 0:
                if not pcd.has_normals():
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                reconstructed, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
                bbox2 = pcd.get_axis_aligned_bounding_box().scale(1.02, pcd.get_axis_aligned_bounding_box().get_center())
                src_mesh = reconstructed.crop(bbox2)
                src_mesh.compute_vertex_normals()
            else:
                raise RuntimeError("Нет данных для обрезки (ни mesh, ни pcd).")

        verts = np.asarray(src_mesh.vertices)
        tris = np.asarray(src_mesh.triangles)
        if tris.size == 0:
            raise RuntimeError("Исходный mesh не содержит треугольников для обрезки.")

        tri_centers_x = verts[tris].mean(axis=1)[:, 0]
        keep_mask = tri_centers_x <= plane_x
        new_tris = tris[keep_mask]
        used_idx = np.unique(new_tris.flatten())
        if used_idx.size == 0:
            print("[WARN] После фильтрации не осталось треугольников — ничего не сохранить.")
            mesh_clipped = o3d.geometry.TriangleMesh()
        else:
            idx_map = {old:int(i) for i, old in enumerate(used_idx)}
            new_vertices = verts[used_idx]
            new_triangles = np.array([[idx_map[int(v)] for v in tri] for tri in new_tris], dtype=np.int32)
            mesh_clipped = o3d.geometry.TriangleMesh()
            mesh_clipped.vertices = o3d.utility.Vector3dVector(new_vertices)
            mesh_clipped.triangles = o3d.utility.Vector3iVector(new_triangles)
            mesh_clipped.compute_vertex_normals()
            if src_mesh.has_vertex_colors():
                cols = np.asarray(src_mesh.vertex_colors)[used_idx]
                mesh_clipped.vertex_colors = o3d.utility.Vector3dVector(cols)

        print("После обрезки: vertices=", len(mesh_clipped.vertices), "triangles=", len(mesh_clipped.triangles))
        print("Has colors:", mesh_clipped.has_vertex_colors(), "Has normals:", mesh_clipped.has_vertex_normals())
        safe_draw([mesh_clipped], window_name="Step 6: Mesh After Clipping")
    except Exception as e:
        print("[ERROR] Ошибка при клиппинге:", e)
        traceback.print_exc()

    # ШАГ 7 
    print_header("Шаг 7: Цветовой градиент и экстремумы")
    try:
        if 'mesh_clipped' in locals() and len(mesh_clipped.vertices) > 0:
            target_mesh = mesh_clipped
        else:
            target_mesh = src_mesh

        if target_mesh is None or len(target_mesh.vertices) == 0:
            print("[ERROR] Нет mesh для шага 7.")
        else:
            mesh_for_color = copy.deepcopy(target_mesh)
            try:
                if mesh_for_color.has_vertex_colors():
                    mesh_for_color.vertex_colors = o3d.utility.Vector3dVector(np.zeros((len(mesh_for_color.vertices), 3)))
            except Exception:
                pass
            axis = GRADIENT_AXIS.lower()
            axis_idx = {"x":0,"y":1,"z":2}.get(axis, 2)
            verts = np.asarray(mesh_for_color.vertices)
            vals = verts[:, axis_idx]
            if vals.ptp() == 0:
                norm_vals = np.zeros_like(vals)
            else:
                norm_vals = (vals - vals.min()) / vals.ptp()
            colors = np.zeros((len(norm_vals),3))
            colors[:,0] = norm_vals
            colors[:,1] = norm_vals
            colors[:,2] = 1.0 - norm_vals
            mesh_for_color.vertex_colors = o3d.utility.Vector3dVector(colors)
            min_idx = int(np.argmin(vals))
            max_idx = int(np.argmax(vals))
            min_coord = verts[min_idx]
            max_coord = verts[max_idx]
            bbox_any = ref_bbox if 'ref_bbox' in locals() else (pcd.get_axis_aligned_bounding_box() if pcd is not None else None)
            sphere_r = 0.01
            if bbox_any is not None:
                sphere_r = max(bbox_any.get_max_extent() * 0.02, 0.01)
            s_min = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
            s_min.paint_uniform_color([1,0,0])
            s_min.translate(min_coord - s_min.get_center())
            s_max = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
            s_max.paint_uniform_color([0,1,0])
            s_max.translate(max_coord - s_max.get_center())
            print("Extrema coords (min,max):", min_coord, max_coord)
            safe_draw([mesh_for_color, s_min, s_max], window_name="Step 7: Gradient + Extrema")
    except Exception as e:
        print("[ERROR] Ошибка в шаге 7:", e)
        traceback.print_exc()

    print_header("")
    print("")
    print("")

if __name__ == "__main__":
    main()
