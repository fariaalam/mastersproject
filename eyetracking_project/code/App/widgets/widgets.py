from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button

class CustomButton(Button):
    '''
    Custom button class.
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CustomDropDown(CustomButton):
    '''
    Custom drop down class.
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dropdown = DropDown()
        self.dropdown_items = {}

        self.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=lambda instance, x: setattr(self, 'text', x))

    def add_item(self, item):
        '''
        Add item to the dropdown.
        If the item is already in the dropdown, it will not be added.
        To compare items, the items need a .text attribute.

        :param item: widget to be added to the dropdown.
        :return: True if the item was added, False otherwise.
        '''
        # test if items have .text attribute
        if not hasattr(item, 'text'):
            return False
        # if item is button/ selectable
        if hasattr(item, 'on_release'):
            item.bind(on_release=lambda btn: self.dropdown.select(btn.text))
        # test if item is already in the dropdown
        if item.text not in self.dropdown_items:
            self.dropdown_items[item.text] = item
            self.dropdown.add_widget(item)
            return True
        return False

    def remove_item(self, item_text):
        '''
        Remove item from the dropdown.
        If the item is not in the dropdown, nothing will happen.

        :param item: .text attribute of the item to be removed from the dropdown.
        :return: True if the item was removed, False otherwise.
        '''
        if item_text in self.dropdown_items:
            self.dropdown.remove_widget(self.dropdown_items[item_text])
            del self.dropdown_items[item_text]
            return True
        return False