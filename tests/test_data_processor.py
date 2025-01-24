import pytest
import os
import json
from src.core.data_processor import DataProcessor

class TestDataProcessor:
    @pytest.fixture
    def processor(self):
        return DataProcessor()
    
    def test_save_to_json(self, processor, tmp_path):
        """Test if JSON file is created correctly"""
        # Save to temporary directory
        output_path = processor.save_to_json(str(tmp_path))
        
        # Check if file exists
        assert os.path.exists(output_path)
        
        # Read and verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check structure
        assert "metadata" in data
        assert "data" in data
        assert "teachers" in data["data"]
        assert "classes" in data["data"]
        assert "students" in data["data"]
    
    def test_class_stats(self, processor):
        """Test class statistics calculation"""
        stats = processor.get_class_stats()
        
        # Check all classes are included
        assert len(stats) == 3
        assert all(class_name in ["3A", "3B", "3C"] for class_name in stats)
        
        # Check statistics for each class
        for class_stats in stats.values():
            assert "total_students" in class_stats
            assert "average_age" in class_stats
            assert "teacher" in class_stats
            assert class_stats["total_students"] > 0
            assert 13 <= class_stats["average_age"] <= 16
    
    def test_get_students_by_class(self, processor):
        """Test retrieving students by class"""
        # Test for each class
        for class_name in ["3A", "3B", "3C"]:
            students = processor.get_students_by_class(class_name)
            assert len(students) > 0
            assert all(s["class"] == class_name for s in students)
    
    def test_get_teacher_info(self, processor):
        """Test retrieving teacher information"""
        # Test for each class
        for class_name in ["3A", "3B", "3C"]:
            teacher = processor.get_teacher_info(class_name)
            assert teacher is not None
            assert teacher["class_responsibility"] == class_name
            assert "name" in teacher
            assert "subject" in teacher

if __name__ == "__main__":
    pytest.main(["-v"])