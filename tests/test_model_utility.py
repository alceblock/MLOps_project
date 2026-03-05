# import os
# import model.model_utility as util

# def test_get_highest_version_empty(tmp_path, monkeypatch):
#     # Usiamo una cartella temporanea per non toccare i tuoi modelli reali
#     base = tmp_path / "models"
#     monkeypatch.setattr(util, "BASE_PATH", str(base))
    
#     assert util.get_highest_version_number() == 0

# def test_get_highest_version_with_folders(tmp_path, monkeypatch):
#     base = tmp_path / "models"
#     base.mkdir()
#     # Creiamo finte cartelle di versioni
#     (base / "model_v_1").mkdir()
#     (base / "model_v_10").mkdir()
#     (base / "model_v_2").mkdir()
    
#     monkeypatch.setattr(util, "BASE_PATH", str(base))
#     assert util.get_highest_version_number() == 10

# def test_new_version_path_creates_dir(tmp_path, monkeypatch):
#     base = tmp_path / "models"
#     monkeypatch.setattr(util, "BASE_PATH", str(base))
    
#     # Se partiamo da zero, deve creare la v1
#     path = util.new_version_path_builder()
#     assert "model_v_1" in path
#     assert os.path.exists(path)

# def test_fallback_to_default_model(tmp_path, monkeypatch):
#     base = tmp_path / "models"
#     monkeypatch.setattr(util, "BASE_PATH", str(base))
    
#     # Non essendoci cartelle, deve tornare il path di CardiffNLP
#     assert util.get_highest_version_model() == util.DEFAULT_MODEL_PATH
