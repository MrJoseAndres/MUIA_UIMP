class Node:
    def __init__(self, items, ocurrences, isPad = False):
        self.items = items.split(":")
        self.ocurrences = ocurrences
        self.children = []
        self.best_children = []
        self.parents = []
        self.locked = True
        self.isPad = isPad

    def add_if_child(self, child):
        if self.isPad:
            self.children.append(child)
            return
        
        for item in child.items:
            if item not in self.items:
                return
        self.children.append(child)

    def add_parent(self, parent):
        if parent.isPad:
            self.parents.append(parent)
            return
        for item in self.items:
            if item not in parent.items:
                return
        self.parents.append(parent)

    def __str__(self):
        return str(self.items) + " : support(" + str(self.ocurrences) + ")"

    def select_parent(self, selected_parent):
        #En vez de eliminar arcos, se crea un arco fuerte
        self.best_parent = selected_parent
        selected_parent.add_best_child(self)

    def add_best_child(self, child):
        self.best_children.append(child)



class ItemSetExtractor:
    FILE = "car.csv"
    SEPARATOR = ","

    def __init__(self, file = FILE, separator = SEPARATOR):
        self.file = file
        self.separator = separator
        self.__load_data__()


    def __combinations__(self, items):
        if not items:
            return []

        first = items[0]
        rest = items[1:]

        combs_without_first = self.__combinations__(rest)

        combs_with_first = []
        for comb_str in combs_without_first:
            combs_with_first.append(first + ":" + comb_str)
        combs_with_first.append(str(first))

        return combs_with_first + combs_without_first

    def __load_data__(self):
        
        
        self.entries = 0
        self.raw_ocurrences_per_len = {} #itemset : ocurrences El itemset tiene el formato "1_a:2_b:3_c"

        print("Counting ocurrences")
        with open(self.file, "r") as file:
            self.header = file.readline().strip().split(self.separator)

            for line in file:
                self.entries += 1
                items = line.strip().split(self.separator)
                real_items = []
                for index, item in enumerate(items):
                    if len(item) > 0:
                        real_items.append(self.header[index] + "_" + item)
                
                if len(real_items) == 0:
                    continue
                itemsets = self.__combinations__(real_items)

                for itemset in itemsets:
                    itemset_length = len(itemset.split(":"))
                    if itemset_length not in self.raw_ocurrences_per_len:
                        self.raw_ocurrences_per_len[itemset_length] = {}
                    if itemset not in self.raw_ocurrences_per_len[itemset_length]:
                        self.raw_ocurrences_per_len[itemset_length][itemset] = 0
                    self.raw_ocurrences_per_len[itemset_length][itemset] += 1

        print("Building tree")
        self.ocurrences = []
        ordered_lengths = sorted(self.raw_ocurrences_per_len.keys(), reverse=True)
        
        for cur_ocurrences, cur_value in self.raw_ocurrences_per_len[ordered_lengths[0]].items():
            self.ocurrences.append(Node(cur_ocurrences, cur_value))

        pad_parent = Node(":" * (ordered_lengths[0] -1), -1, True)
        self.ocurrences.append(pad_parent)

        last_ocurrences = self.ocurrences
        ocurrences_per_level = {} #No contiene los nodos de mayor nivel. Se usará para eliminar arcos redundantes.
        for current_length in ordered_lengths[1:]:
            new_ocurrences = []
            ocurrences_per_level[current_length] = []
            no_match = []
            for cur_ocurrences, cur_value in self.raw_ocurrences_per_len[current_length].items():
                ocurrence = Node(cur_ocurrences, cur_value)
                matches = 0
                
                for last_ocurrence in last_ocurrences:
                    last_ocurrence.add_if_child(ocurrence)
                    ocurrence.add_parent(last_ocurrence)
                    matches += 1
                if matches == 0:
                    no_match.append(ocurrence)

                ocurrences_per_level[current_length].append(ocurrence)
                new_ocurrences.append(ocurrence)

            if len(no_match) > 0:
                current_level_pad = Node(":" * (current_length -1), -1, True)
                for no_match_item in no_match:
                    no_match_item.add_parent(current_level_pad)
                    current_level_pad.add_child(no_match_item)
                new_ocurrences.append(current_level_pad)
                ocurrences_per_level[current_length].append(current_level_pad)
            
            last_ocurrences = new_ocurrences

        print("Selecting best parents")
        ordered_lengths = sorted(ocurrences_per_level.keys())

        for current_length in ordered_lengths:
            for keyset in ocurrences_per_level[current_length]:
                self.select_best_parent(keyset)


    def select_best_parent(self, node):
        parents = {}
        for parent in node.parents:
            parent_ocurrences = parent.ocurrences
            if parent_ocurrences not in parents:
                parents[parent_ocurrences] = []
            parents[parent_ocurrences].append(parent)
        
        max_ocurrences = max(parents.keys())
        selected_parent = parents[max_ocurrences][0]
        node.select_parent(selected_parent)
    

    def __get_max_itemsets__(self, node, support):
        if node.ocurrences >= support:
            return [str(node)]
        
        max_itemsets = []
        for child in node.best_children:
            max_itemsets.append(self.__get_max_itemsets__(child, support))

        return max_itemsets
    
    def __get_closed_itemsets__(self, node, support):
        closed_itemsets = []
        #check if node is closed
        if node.ocurrences >= support:
            closed = True
            for parent in node.parents:
                if parent.ocurrences == node.ocurrences:
                    closed = False
                    break
            if closed:
                closed_itemsets.append(str(node))
        
        for child in node.best_children:
            for child_closed_itemset in self.__get_closed_itemsets__(child, support):
                if len(child_closed_itemset) > 0:
                    closed_itemsets.append(child_closed_itemset)

        return closed_itemsets
        


    def maximal_itemsets(self, minimum_support):
        minimum_occurrences = self.entries * minimum_support

        maximal_itemsets = []
        for itemset in self.ocurrences:
            for max_itemset in self.__get_max_itemsets__(itemset, minimum_occurrences):
                if len(max_itemset) > 0:
                    maximal_itemsets.append(max_itemset)
        return maximal_itemsets
    

    def closed_itemsets(self, minimum_support):
        minimum_occurrences = self.entries * minimum_support

        closed_itemsets = []
        for itemset in self.ocurrences:
            for closed_itemset in self.__get_closed_itemsets__(itemset, minimum_occurrences):
                if len(closed_itemset) > 0:
                    closed_itemsets.append(closed_itemset)

        return closed_itemsets



def main():
    file = input("Introduce el archivo de datos (nombre sin extensión): ") + ".csv"
    separator = input("Introduce el separador del csv (por ejemplo ,): ")
    fpg = ItemSetExtractor(file, separator)
    option = None
    while option != "0":
        print("====================================")
        print("1. Calcular ítemsets máximos")
        print("2. Calcular ítemsets cerrados")
        print("0. Salir")
        print("====================================")
        option = input("Introduce una opción: ")
        if option == "1" or option == "2":
            min_support = -1
            while min_support < 0 or min_support > 1:
                try:
                    min_support = float(input("Introduce el soporte mínimo (valor entre 0 y 1): "))
                except:
                    print("Valor no válido")
                    min_support = -1

            if option == "1":
                maximal_itemsets = fpg.maximal_itemsets(min_support)
                print(maximal_itemsets)
                print("--------------------")
                print(f"Total: {len(maximal_itemsets)}")
            elif option == "2":
                closed_itemsets = fpg.closed_itemsets(min_support)
                print(closed_itemsets)
                print("--------------------")
                print(f"Total: {len(closed_itemsets)}")

        elif option == "0":
            print("Saliendo...")
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()