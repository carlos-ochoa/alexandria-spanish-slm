import wikipedia
from typing import List, Optional


class WikipediaCategoryNavigator:
    """
    A class to navigate Wikipedia categories and retrieve subcategories.
    """

    def __init__(self, language: str = 'es'):
        """
        Initialize the Wikipedia category navigator.

        Args:
            language: Wikipedia language code (default: 'es')
        """
        self.language = language
        wikipedia.set_lang(language)

    def get_subcategories(self, category_name: str) -> List[str]:
        """
        Get all subcategories for a given Wikipedia category.

        Args:
            category_name: Name of the category (e.g., 'Category:Science' or just 'Science')

        Returns:
            List of subcategory names
        """
        # Ensure category name has the 'Category:' prefix
        if not category_name.startswith('Category:'):
            category_name = f'Category:{category_name}'

        try:
            # Get the category page
            category_page = wikipedia.page(category_name, auto_suggest=False)

            # Extract subcategories from the page content
            # Note: The wikipedia package doesn't have direct category API support,
            # so we'll need to parse categories from the page
            subcategories = []

            # Get categories this page belongs to
            categories = category_page.categories

            # Filter for subcategories (categories that start with the parent category name)
            for cat in categories:
                if cat.startswith('Category:'):
                    subcategories.append(cat)

            return subcategories

        except wikipedia.exceptions.PageError:
            print(f"Category '{category_name}' not found.")
            return []
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for '{category_name}': {e.options}")
            return []
        except Exception as e:
            print(f"Error retrieving subcategories: {str(e)}")
            return []

    def get_category_members(self, category_name: str, limit: Optional[int] = None) -> List[str]:
        """
        Get member pages of a category.

        Args:
            category_name: Name of the category
            limit: Maximum number of members to retrieve (None for all)

        Returns:
            List of page titles in the category
        """
        if not category_name.startswith('Category:'):
            category_name = f'Category:{category_name}'

        try:
            # Search for pages in the category
            search_results = wikipedia.search(category_name, results=limit or 10)
            return search_results
        except Exception as e:
            print(f"Error retrieving category members: {str(e)}")
            return []

    def set_language(self, language: str):
        """
        Change the Wikipedia language.

        Args:
            language: Wikipedia language code (e.g., 'es' for Spanish, 'en' for English)
        """
        self.language = language
        wikipedia.set_lang(language)

import wikipediaapi

wiki = wikipediaapi.Wikipedia(user_agent='Alexandria-Spanish-SLM', language='es')

categoria = wiki.page("Categoría:Años_0_en_el_Imperio_romano")
print(list(categoria.categorymembers.keys()))

categoria = wiki.page("Categoría:Años_en_el_Imperio_romano")
print(list(categoria.categorymembers.keys()))
