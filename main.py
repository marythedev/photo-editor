import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons, TextBox

def img_difference(current, new):
    """
    shows 2 images side by side to display the difference made by the operation
    """
    figure, (axis_current, axis_new) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)

    image1 = axis_current.imshow(cv2.cvtColor(current, cv2.COLOR_BGR2RGB))
    axis_current.set_title('Original (current)')
    axis_current.axis('off')

    image2 = axis_new.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))
    axis_new.set_title('Preview (new)')
    axis_new.axis('off')
    return figure, (axis_current, axis_new), (image1, image2)

def adjust_brightness(img):
    img_current = img.copy()
    img_new = img.copy()

    figure, (axis_current, axis_new), (image1, image2) = img_difference(img_current, img_new)
    figure.suptitle('Brightness Adjustment', fontsize=16, fontweight='bold', y=0.95)

    # slider for brightness adjustment
    slider_position = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_position, 'Brightness', -100, 100, valinit=0)

    def update(val):
        value = int(slider.val)
        new_img = cv2.convertScaleAbs(img_current, alpha=1, beta=value)     # brightness adjustment
        img_new[:] = new_img
        image2.set_data(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
        figure.canvas.draw_idle()

    slider.on_changed(update)

    # buttons to complete or disregard operation
    done = False

    done_btn_position = plt.axes([0.3, 0.02, 0.2, 0.05])
    done_btn = Button(done_btn_position, 'Done')
    def on_done(event):
        nonlocal done
        done = True
        plt.close()
    done_btn.on_clicked(on_done)
    
    cancel_btn_position = plt.axes([0.6, 0.02, 0.2, 0.05])
    cancel_btn = Button(cancel_btn_position, 'Cancel')
    def on_cancel(event):
        nonlocal done
        done = False
        plt.close()
    cancel_btn.on_clicked(on_cancel)
    
    plt.show()
    return ("Success", img_new, slider.val) if done else ("Cancelled", img_current, 0)

def adjust_contrast(img):
    img_current = img.copy()
    img_new = img.copy()

    figure, (axis_current, axis_new), (image1, image2) = img_difference(img_current, img_new)
    figure.suptitle('Contrast Adjustment', fontsize=16, fontweight='bold', y=0.95)

    # slider for contrast adjustment
    slider_position = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_position, 'Contrast', 0.0, 2.0, valinit=1.0)

    def update(val):
        alpha = float(val)
        new_img = cv2.convertScaleAbs(img_current, alpha=alpha, beta=0)   # contrast adjustment
        img_new[:] = new_img
        image2.set_data(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
        figure.canvas.draw_idle()

    slider.on_changed(update)


    # buttons to complete or disregard operation
    done = False

    done_btn_position = plt.axes([0.3, 0.02, 0.2, 0.05])
    done_btn = Button(done_btn_position, 'Done')
    def on_done(event):
        nonlocal done
        done = True
        plt.close()
    done_btn.on_clicked(on_done)

    cancel_btn_position = plt.axes([0.6, 0.02, 0.2, 0.05])
    cancel_btn = Button(cancel_btn_position, 'Cancel')
    def on_cancel(event):
        nonlocal done
        done = False
        plt.close()
    cancel_btn.on_clicked(on_cancel)
    
    plt.show()
    return ("Success", img_new, slider.val) if done else ("Cancelled", img_current, 1.0)

def convert_to_grayscale(img):
    img_current = img.copy()
    # check if the image is already grayscale
    if len(img_current.shape) == 2:
        img_new = img_current
    else:
        img_new = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)

    figure, (axis_current, axis_new), (image1, image2) = img_difference(img_current, img_new)
    figure.suptitle('Convert to Grayscale', fontsize=16, fontweight='bold', y=0.95)

    # button to close the window
    ok_btn_position = plt.axes([0.6, 0.02, 0.2, 0.05])
    ok_btn = Button(ok_btn_position, 'Okay')
    def on_done(event):
        plt.close()
    ok_btn.on_clicked(on_done)

    plt.show()
    return img_new

def add_padding(img):
    img_current = img.copy()
    img_new = img.copy()

    padding_size = 50
    border_type_name = 'Constant'
    ratio = 'Rectangle'
    custom_ratio = (4, 5)

    figure, (axis_current, axis_new), (image1, image2) = img_difference(img_current, img_new)
    figure.suptitle('Add padding', fontsize=16, fontweight='bold', y=0.95)

    # slider to adjust padding size
    slider_position = plt.axes([0.15, 0.22, 0.7, 0.03])
    slider = Slider(slider_position, 'Padding Size', 0, 500, valinit=padding_size)

    # ratio buttons
    square_btn_position = plt.axes([0.05, 0.16, 0.15, 0.05])
    square_btn = Button(square_btn_position, 'Square')
    rectangle_btn_position = plt.axes([0.25, 0.16, 0.15, 0.05])
    rectangle_btn = Button(rectangle_btn_position, 'Rectangle')
    custom_btn_position = plt.axes([0.45, 0.16, 0.15, 0.05]) 
    custom_btn = Button(custom_btn_position, 'Custom Ratio')

    # custom ratio inputs
    ratio_input1_position = plt.axes([0.65, 0.16, 0.05, 0.05])
    ratio_input1 = TextBox(ratio_input1_position, '', initial=str(custom_ratio[0]))
    ratio_input2_position = plt.axes([0.71, 0.16, 0.05, 0.05])
    ratio_input2 = TextBox(ratio_input2_position, '', initial=str(custom_ratio[1]))
    ratio_input1_position.set_visible(False)
    ratio_input2_position.set_visible(False)

    # border type buttons
    border1_btn_position = plt.axes([0.05, 0.10, 0.15, 0.05])
    border1_btn = Button(border1_btn_position, 'Constant')
    border2_btn_position = plt.axes([0.25, 0.10, 0.15, 0.05])
    border2_btn = Button(border2_btn_position, 'Reflect')
    border3_btn_position = plt.axes([0.45, 0.10, 0.15, 0.05]) 
    border3_btn = Button(border3_btn_position, 'Replicate')

    border_map = {
        'Constant': cv2.BORDER_CONSTANT,
        'Reflect': cv2.BORDER_REFLECT,
        'Replicate': cv2.BORDER_REPLICATE
    }

    def apply_padding(val=None):
        nonlocal img_new, custom_ratio
        h, w = img_current.shape[:2]
        padding = int(slider.val)

        extra = 0
        if border_type_name == 'Reflect':
            extra = padding // 2
        elif border_type_name == 'Replicate':
            extra = padding


        if ratio == 'Square':
            target = max(h, w)
            padding_horiz = (target - h) // 2
            padding_vert = (target - w) // 2
            padding_top = padding_vert + padding + extra
            padding_bottom = padding_vert + padding + extra
            padding_left = padding_horiz + padding + extra
            padding_right = padding_horiz + padding + extra


        elif ratio == 'Rectangle':
            padding_top = padding_bottom = padding_left = padding_right = padding + extra
        elif ratio == 'Custom Ratio':
            try:
                num = int(ratio_input1.text)
                den = int(ratio_input2.text)
                if num <= 0 or den <= 0:
                    raise ValueError
                custom_ratio = (num, den)
                target_ratio = custom_ratio[0] / custom_ratio[1]
                current_ratio = w / h
                if current_ratio < target_ratio:
                    new_w = int(h * target_ratio)
                    padding_left = (new_w - w) // 2 + padding + extra
                    padding_right = padding_left
                    padding_top = padding + extra
                    padding_bottom = padding + extra
                else:
                    new_h = int(w / target_ratio)
                    padding_top = (new_h - h) // 2 + padding + extra
                    padding_bottom = padding_top
                    padding_left = padding + extra
                    padding_right = padding + extra
            except:
                padding_top = padding_bottom = padding_left = padding_right = padding + extra

        img_new = cv2.copyMakeBorder(
            img_current, padding_top, padding_bottom, padding_left, padding_right,
            borderType=border_map[border_type_name], value=(0, 0, 0)
        )
        image2.set_data(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
        figure.canvas.draw_idle()

    def on_square(event):
        nonlocal ratio
        ratio = 'Square'
        ratio_input1_position.set_visible(False)
        ratio_input2_position.set_visible(False)
        apply_padding()

    def on_rectangle(event):
        nonlocal ratio
        ratio = 'Rectangle'
        ratio_input1_position.set_visible(False)
        ratio_input2_position.set_visible(False)
        apply_padding()

    def on_custom(event):
        nonlocal ratio
        ratio = 'Custom Ratio'
        ratio_input1_position.set_visible(True)
        ratio_input2_position.set_visible(True)
        apply_padding()

    def on_ratio_change(_):
        if ratio == 'Custom Ratio':
            apply_padding()

    def set_border_type(name):
        nonlocal border_type_name
        border_type_name = name
        apply_padding()

    slider.on_changed(apply_padding)

    square_btn.on_clicked(on_square)
    rectangle_btn.on_clicked(on_rectangle)
    custom_btn.on_clicked(on_custom)

    ratio_input1.on_submit(on_ratio_change)
    ratio_input2.on_submit(on_ratio_change)

    border1_btn.on_clicked(lambda event: set_border_type('Constant'))
    border2_btn.on_clicked(lambda event: set_border_type('Reflect'))
    border3_btn.on_clicked(lambda event: set_border_type('Replicate'))


    apply_padding()

    # buttons to complete or disregard operation
    done = False

    done_btn_position = plt.axes([0.05, 0.02, 0.15, 0.05])

    done_btn = Button(done_btn_position, 'Done')
    def on_done(event):
        nonlocal done
        done = True
        plt.close()
    done_btn.on_clicked(on_done)
    
    cancel_btn_position = plt.axes([0.25, 0.02, 0.15, 0.05])
    cancel_btn = Button(cancel_btn_position, 'Cancel')
    def on_cancel(event):
        nonlocal done
        done = False
        plt.close()
    cancel_btn.on_clicked(on_cancel)
    
    plt.show()
    return ("Success", img_new, slider.val, border_type_name, ratio) if done else ("Cancelled", img_current, padding_size, border_type_name, ratio)

def apply_threshold(img):
    img_current = img.copy()

    # convert to grayscale if the image is not 
    if len(img_current.shape) == 3:
        img_current = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)

    img_new = img_current.copy()

    figure, (axis_current, axis_new), (image1, image2) = img_difference(img_current, img_new)
    figure.suptitle('Apply Thresholding (binary or inverse)', fontsize=16, fontweight='bold', y=0.95)

    # radio buttons for threshold type
    radio_position = plt.axes([0.15, 0.15, 0.25, 0.1])
    radio_position.axis('off')
    radio = RadioButtons(radio_position, ('Binary', 'Binary Inverted'))
    threshold_type = cv2.THRESH_BINARY

    threshold_value = 127

    def update():
        _, threshold_img = cv2.threshold(img_current, threshold_value, 255, threshold_type)
        img_new[:] = threshold_img
        image2.set_data(cv2.cvtColor(img_new, cv2.COLOR_GRAY2RGB) if len(img_new.shape) == 2 else img_new)
        figure.canvas.draw_idle()

    def on_radio(label):
        nonlocal threshold_type
        if label == 'Binary':
            threshold_type = cv2.THRESH_BINARY
        else:
            threshold_type = cv2.THRESH_BINARY_INV
        update()

    radio.on_clicked(on_radio)

    update()

    # buttons to complete or disregard operation
    done = False

    done_btn_position = plt.axes([0.3, 0.02, 0.2, 0.05])
    done_btn = Button(done_btn_position, 'Done')
    def on_done(event):
        nonlocal done
        done = True
        plt.close()
    done_btn.on_clicked(on_done)
    
    cancel_btn_position = plt.axes([0.6, 0.02, 0.2, 0.05])
    cancel_btn = Button(cancel_btn_position, 'Cancel')
    def on_cancel(event):
        nonlocal done
        done = False
        plt.close()
    cancel_btn.on_clicked(on_cancel)

    plt.show()

    if done:
        option = 'BINARY' if threshold_type == cv2.THRESH_BINARY else 'BINARY_INV'
        return "Success", img_new, option
    else:
        return "Cancelled", img_current, "None"

def blend_with_image(img1):
    img_current = img1.copy()

    # check if the image is grayscale
    if len(img_current.shape) == 2:
        img_current = cv2.cvtColor(img_current, cv2.COLOR_GRAY2BGR)

    img_new = img_current.copy()

    # image to blend with
    PATH_TO_IMG2 = input("\nEnter the image file name to blend the original image with (i.e. overlay_image.jpg): ")
    img2 = cv2.imread(PATH_TO_IMG2)
    if img2 is None:
        print("Failed to load image.\n")
        return "Failed", img1, None
    
    # making sure both images are of the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    figure, (axis_current, axis_new), (image1, image2) = img_difference(img_current, img_new)
    figure.suptitle('Blend with another image', fontsize=16, fontweight='bold', y=0.95)

    # slider for alpha adjustment
    slider_position = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_position, 'Alpha', 0.0, 1.0, valinit=0.35)

    def update(val):
        alpha = float(val)
        blend = (1 - alpha) * img_current + alpha * img2
        img_new[:] = np.clip(blend, 0, 255).astype(np.uint8)
        image2.set_data(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
        figure.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)

    # buttons to complete or disregard operation
    done = False

    done_btn_position = plt.axes([0.3, 0.02, 0.2, 0.05])
    done_btn = Button(done_btn_position, 'Done')
    def on_done(event):
        nonlocal done
        done = True
        plt.close()
    done_btn.on_clicked(on_done)
    
    cancel_btn_position = plt.axes([0.6, 0.02, 0.2, 0.05])
    cancel_btn = Button(cancel_btn_position, 'Cancel')
    def on_cancel(event):
        nonlocal done
        done = False
        plt.close()
    cancel_btn.on_clicked(on_cancel)
    
    plt.show()
    return ("Success", img_new, slider.val) if done else ("Cancelled", img_current, 0.35)

def history_of_operations(actions):
    print("\nHISTORY OF OPERATIONS:")
    if not actions:
        print("No operations performed yet.")
    else:
        for i, act in enumerate(actions):
            print(f"{i+1}: {act}")

def main():
    img = None
    while img is None:
        PATH_TO_IMG1 = input("Enter the name to the image file (i.e. image.jpg): ")
        img = cv2.imread(PATH_TO_IMG1)
        if (img is None):
            print("Failed to load image, try another file.\n")

    history = [img.copy()]
    actions = []

    while True:
        print("\n=== Mini Photo Editor ===")
        print("1. Adjust Brightness")
        print("2. Adjust Contrast")
        print("3. Convert to Grayscale")
        print("4. Add Padding (interactive)")
        print("5. Apply Thresholding (binary or inverse)")
        print("6. Blend with Another Image (manual alpha)")
        print("7. Undo Last Operation")
        print("8. View History of Operations")
        print("9. Save and Exit")
        choice = input("Choose an option (1-9): ")

        if choice == '1':
            status, img, brightness = adjust_brightness(img)
            if status == "Success":
                actions.append(f"brightness adjustment by {int(brightness)}")
                history.append(img.copy())
        elif choice == '2':
            status, img, contrast = adjust_contrast(img)
            if status == "Success":
                actions.append(f"contrast adjustment by {contrast:.2f}")
                history.append(img.copy())
        elif choice == '3':
            img = convert_to_grayscale(img)
            actions.append("converted to grayscale")
            history.append(img.copy())
        elif choice == '4':
            status, img, padding, border_type, ratio = add_padding(img)
            if status == "Success":
                actions.append(f"added padding of size {padding} with {border_type} border type and {ratio} ratio")
                history.append(img.copy())
        elif choice == '5':
            status, img, option = apply_threshold(img)
            if status == "Success":
                actions.append(f"applied {option} thresholding")
                history.append(img.copy())
        elif choice == '6':
            status, img, alpha = blend_with_image(img)
            if status == "Success":
                actions.append(f"blended with another image at alpha {alpha:.2f}")
                history.append(img.copy())
        elif choice == '7':
            if len(history) > 1:
                history.pop()
                img = history[-1].copy()
                actions.append("undo")
                print("\nLAST OPERATION WAS UNDONE SUCCESSFULLY.")
            else:
                print("\nNO ACTIONS TO UNDO.")
        elif choice == '8':
            history_of_operations(actions)
        elif choice == '9':
            history_of_operations(actions)
            save = input("Save final image? (y/n): ").strip().lower()
            if save == 'y':
                print("Image will be saved in .jpg format.")
                fname = input("\nEnter filename to save (e.g. edited_image): ")
                cv2.imwrite(f'{fname}.jpg', img)
                print(f"Image successfully saved as {fname}")
            break
        else:
            print("\nOPTION DOESN'T EXIST, PLEASE ENTER THE NUMBER FROM THE MENU")

if __name__ == "__main__":
    main()
