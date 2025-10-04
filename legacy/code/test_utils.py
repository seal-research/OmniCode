def test_get_modified_functions(tmp_path):
    """Test get_modified_functions utility"""
    from pathlib import Path
    import os
    from git import Repo
    
    # Create a temporary git repo
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    repo = Repo.init(repo_path)
    
    # Create a Python file with some functions
    test_file = repo_path / "test.py"
    initial_content = '''
def function1():
    return 1

class TestClass:
    def method1(self):
        return "hello"
        
    def method2(self):
        return "world"
'''
    test_file.write_text(initial_content)
    
    # Add and commit the file
    repo.index.add(['test.py'])
    base_commit = repo.index.commit("Initial commit")
    
    # Modify the file
    modified_content = '''
def function1():
    return 2  # Modified

class TestClass:
    def method1(self):
        return "hi"  # Modified
        
    def method2(self):
        return "world"
'''
    test_file.write_text(modified_content)
    
    # Create diff
    repo.index.add(['test.py'])
    diff = repo.git.diff(base_commit.hexsha)
    
    # Get modified functions
    from utils import get_modified_functions
    modified_funcs = get_modified_functions(diff, repo_path, base_commit.hexsha)
    
    # Check results
    expected = [
        'test.py:function1',
        'test.py:TestClass.method1'
    ]
    
    assert sorted(modified_funcs) == sorted(expected), \
        f"Expected {expected}, but got {modified_funcs}"

