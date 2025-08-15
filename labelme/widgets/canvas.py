import contextlib

import imgviz
from loguru import logger
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtGui import QMouseEvent
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMessageBox

import labelme.ai
import labelme.utils
from labelme import QT5
from labelme.shape import Shape

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    mouseMoved = QtCore.Signal(QtCore.QPointF)
    mousePressed = QtCore.Signal(QMouseEvent)  # 定义信号，用于判断是否是在多边形标签列表中做的操作
    mouseReleased = QtCore.Signal(QMouseEvent)  # 定义信号，用于判断是否是在多边形标签列表中做的操作
    editingSaveEnable = QtCore.Signal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(self.double_click)
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        self._ai_model = None

        self.drawing_enabled = False  # 新增属性
        self._select_mode = False
        self.original_image = None  # 新增属性，用于存储原图

        self.allow_drag = False  # 添加一个是否允许拖拽的判断值
        self.current_filename = None  # 储存当前文件名

        self.selecting = False  # 是否正在框选
        self.select_start = None  # 框选起始点
        self.select_rect = None  # 框选矩形

        # 定义可用的标签类型
        self.keypoint_labels = ["keypoint"]  # 关键点标签
        self.line_labels = ["line", "entrance_line"]  # 其他标签

    def setCursor(self, cursor):
        """设置光标样式"""
        self._cursor = cursor  # 保存当前光标样式
        QtWidgets.QApplication.setOverrideCursor(cursor)  # 设置全局光标样式

    def setDrawingEnabled(self, enabled):
        self.drawing_enabled = enabled
        # if enabled:
        #     self.mode = self.CREATE  # 切换到绘制模式
        # else:
        #     self.mode = self.EDIT  # 切换到编辑模式
        self.update()  # 更新画布

    def setSelectMode(self, enable):
        """设置是否为选择模式"""
        self._select_mode = enable
        # self.setCursor(QtCore.Qt.ArrowCursor if enable else QtCore.Qt.OpenHandCursor)
    
    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_mask",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def initializeAiModel(self, name):
        if name not in [model.name for model in labelme.ai.MODELS]:
            raise ValueError("Unsupported ai model: %s" % name)
        model = [model for model in labelme.ai.MODELS if model.name == name][0]

        if self._ai_model is not None and self._ai_model.name == model.name:
            logger.debug("AI model is already initialized: %r" % model.name)
        else:
            logger.debug("Initializing AI model: %r" % model.name)

            class LoggerIO:
                def write(self, message: str):
                    if message := message.strip():
                        logger.debug(message)

                def flush(self):
                    pass

            # NOTE: gdown.download uses sys.stderr, so redirect it to logger.debug
            with contextlib.redirect_stderr(new_target=LoggerIO()):
                self._ai_model = model()

        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

        self._ai_model.set_image(
            image=labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        )

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        # return self.mode == self.CREATE
        return self.drawing_enabled and self.mode == self.CREATE  # 修改绘制条件

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def ori_mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)

        self.prevMovePoint = pos
        self.restoreCursor()

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        # Polygon drawing.
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.point_labels = [1]
                self.line.close()
            assert len(self.line.points) == len(self.line.point_labels)
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                # self.overrideCursor(CURSOR_MOVE)
                # self.boundedMoveShapes(self.selectedShapes, pos)
                # self.repaint()
                # self.movingShape = True
                pass
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            index_edge = shape.nearestEdge(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr(
                        "Click & Drag to move point\n"
                        "ALT + SHIFT + Click to delete point"
                    )
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("ALT + Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def ori_mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        self.current.addPoint(
                            self.line.points[1],
                            label=self.line.point_labels[1],
                        )
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[-1]
                        if ev.modifiers() & QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(
                        shape_type="points"
                        if self.createMode in ["ai_polygon", "ai_mask"]
                        else self.createMode
                    )
                    self.current.addPoint(pos, label=0 if is_shift_pressed else 1)
                    if self.createMode == "point":
                        self.finalise()
                    elif (
                        self.createMode in ["ai_polygon", "ai_mask"]
                        and ev.modifiers() & QtCore.Qt.ControlModifier
                    ):
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        if (
                            self.createMode in ["ai_polygon", "ai_mask"]
                            and is_shift_pressed
                        ):
                            self.line.point_labels = [0, 0]
                        else:
                            self.line.point_labels = [1, 1]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge() and ev.modifiers() == QtCore.Qt.AltModifier:
                    self.addPointToEdge()
                elif self.selectedVertex() and ev.modifiers() == (
                    QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier
                ):
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
                self.update()
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos
            self.update()

    def ori_mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if self.shapesBackups[-1][index].points != self.shapes[index].points:
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and (
            (self.current and len(self.current) > 2)
            or self.createMode in ["ai_polygon", "ai_mask"]
        )

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return

        if (
            self.createMode == "polygon" and self.canCloseShape()
        ) or self.createMode in ["ai_polygon", "ai_mask"]:
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        # print("=====Parent:", self.parent())
        # 创建一个新的 QMouseEvent 对象（根据需要设置参数）
        event = QMouseEvent(QtCore.QEvent.MouseButtonPress, QtCore.QPoint(0, 0), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        # 发射信号
        self.mousePressed.emit(event)
        
        # 调用 mouseReleaseEvent 和 mousePressEvent，传递 event 对象
        self.selectionChanged.emit(shapes)
        self.update()
        
        # 这里可以根据需要创建另一个 QMouseEvent 对象
        event = QMouseEvent(QtCore.QEvent.MouseButtonRelease, QtCore.QPoint(0, 0), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        # 发射信号
        self.mouseReleased.emit(event)
        # print("Over=====")

    # def selectShapePoint(self, point, multiple_selection_mode):
    #     """Select the first shape created which contains this point."""
    #     if self.selectedVertex():  # A vertex is marked for selection.
    #         index, shape = self.hVertex, self.hShape
    #         shape.highlightVertex(index, shape.MOVE_VERTEX)
    #     else:
    #         for shape in reversed(self.shapes):
    #             if self.isVisible(shape) and shape.containsPoint(point):
    #                 self.setHiding()
    #                 if shape not in self.selectedShapes:
    #                     if multiple_selection_mode:
    #                         self.selectionChanged.emit(self.selectedShapes + [shape])
    #                     else:
    #                         self.selectionChanged.emit([shape])
    #                     self.hShapeIsSelected = False
    #                 else:
    #                     self.hShapeIsSelected = True
    #                 self.calculateOffsets(point)
    #                 return
    #     self.deSelectShape()

    def selectShapePoint(self, point, multiple_selection_mode):
        # 检查是否是 slot 图
        is_slot = self.current_filename and "slot" in self.current_filename.lower()
        is_2dod = self.current_filename and "2d-od" in self.current_filename.lower()
        if is_slot or is_2dod:
            # 对于 slot 图，找到距离点击点最近的形状
            min_distance = float('inf')
            closest_shape = None
            
            for shape in reversed(self.shapes):
                if self.isVisible(shape):
                    distance = self.distanceToShape(point, shape)
                    if distance < min_distance:
                        min_distance = distance
                        closest_shape = shape
            
            if closest_shape is not None:
                self.setHiding()
                if closest_shape not in self.selectedShapes:
                    if multiple_selection_mode:
                        self.selectionChanged.emit(self.selectedShapes + [closest_shape])
                    else:
                        self.selectionChanged.emit([closest_shape])
                    self.hShapeIsSelected = False
                else:
                    self.hShapeIsSelected = True
                self.calculateOffsets(point)
                return
        
        # 获取点击点的像素值
        pixel_value = self.getPixelValue(point)  # 该方法可以获取像素值
        # 根据像素值判断类别
        category = self.getCategoryFromPixel(pixel_value)  # 该方法可以判断类别
        # 遍历该类别的形状
        for shape in reversed(self.shapes):
            if self.isVisible(shape) and shape.label == category:
                if shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(self.selectedShapes + [shape])
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return

        # 如果没有找到包含点击点的形状，取消选择
        self.deSelectShape()

    def distanceToShape(self, point, shape):
        """计算点到形状的最小距离"""
        min_distance = float('inf')
        points = shape.points
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            # 计算点到线段的距离
            distance = self.pointToLineDistance(point, p1, p2)
            min_distance = min(min_distance, distance)
        return min_distance

    def pointToLineDistance(self, point, line_p1, line_p2):
        """计算点到线段的距离"""
        x, y = point.x(), point.y()
        x1, y1 = line_p1.x(), line_p1.y()
        x2, y2 = line_p2.x(), line_p2.y()
        
        # 计算点到线段的距离
        if x1 == x2 and y1 == y2:
            return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
        
        # 计算点到线段的投影点
        dx = x2 - x1
        dy = y2 - y1
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        
        if t < 0:
            return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
        elif t > 1:
            return ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
        else:
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            return ((x - proj_x) ** 2 + (y - proj_y) ** 2) ** 0.5

    def getPixelValue(self, point):
        """
        获取指定点的像素值
        Args:
            point: QPointF 对象，表示要获取像素值的坐标点
        Returns:
            tuple: (R, G, B) 像素值，如果无法获取则返回 None
        """
        if not self.pixmap:
            return None

        # 将坐标点转换为相对于 pixmap 的坐标
        # 考虑缩放和偏移的影响
        scaled_point = QtCore.QPointF(
            # point.x() / self.scale,
            # point.y() / self.scale
            point.x(),
            point.y()
        )
        
        # 获取偏移量
        # offset = self.offsetToCenter()
        
        # 计算实际像素坐标
        # pixel_x = int(scaled_point.x() - offset.x())
        # pixel_y = int(scaled_point.y() - offset.y())
        pixel_x = int(scaled_point.x())
        pixel_y = int(scaled_point.y())
        
        # 检查坐标是否在 pixmap 范围内
        if (pixel_x < 0 or pixel_x >= self.pixmap.width() or 
            pixel_y < 0 or pixel_y >= self.pixmap.height()):
            return None
        
        # 将 pixmap 转换为 QImage 以获取像素值
        image = self.pixmap.toImage()
        
        # 获取像素值
        pixel = image.pixel(pixel_x, pixel_y)
        
        # 从像素值中提取 RGB 分量
        red = QtGui.qRed(pixel)
        green = QtGui.qGreen(pixel)
        blue = QtGui.qBlue(pixel)

        # print("========canvas.getPixelValue.xy.rgb=========", pixel_x, pixel_y, red, green, blue)
        
        return (red, green, blue)
            

    def getCategoryFromPixel(self, pixel_value):
        # 根据像素值判断类别
        cx_color_dict_OurVersion = {
            "Background": [70, 70, 70],
            "Road": [128, 64, 128],
            "Lane_line": [240, 240, 240],
            "Parking_line": [70, 120, 120],
            "Parking_slot": [70, 12, 120],
            "Arrow": [70, 120, 12],
            "Crosswalk_line": [0, 120, 120],
            "No_parking_sign_line": [200, 120, 120],
            "Speed_bump": [70, 200, 120],
            "Parking_lock_open": [70, 120, 200],
            "Parking_lock_closed": [100, 0, 0],
            "Traffic_cone": [130, 100, 10], # 1
            "Parking_rod": [250, 120, 120],
            "Limiter_pole": [70, 0, 250],
            "Pillar": [250, 170, 160], # 1
            "Immovable_obstacle": [150, 120, 90], # 1
            "Person": [220, 20, 60],
            "Car": [0, 0, 142],
            "self_car": [20, 180,80],
            "Curb":  [140, 100, 100],
            "Movable_obstacle": [230, 150, 140], #1
            "Guide_line": [160, 160, 160],
            "Center_lane": [170, 10, 10],
            "Cover": [170, 170, 170],
            "sewer": [100, 20, 10]
        }
        # 这里需要根据实际情况实现判断类别的逻辑
        
        if pixel_value is None:
            return None

        # 计算当前像素值与所有类别的颜色值的欧氏距离
        min_distance = float('inf')
        closest_category = None
        
        for category, color in cx_color_dict_OurVersion.items():
            # 计算欧氏距离
            distance = sum((pixel_value[i] - color[i]) ** 2 for i in range(3))
            
            # 更新最小距离和对应的类别
            if distance < min_distance:
                min_distance = distance
                closest_category = category

        # 设置一个阈值，如果距离太大，可能不属于任何类别
        threshold = 100  # 这个阈值可以根据实际情况调整
        if min_distance > threshold:
            return None

        return closest_category

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveVertex_2DOD(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        
        # 计算移动的偏移量
        dx = pos.x() - point.x()
        dy = pos.y() - point.y()
        
        # 记录修改点原来的坐标
        original_x = point.x()
        original_y = point.y()
        
        # 记录是否已经修改过x坐标和y坐标的点
        x_modified = False
        y_modified = False
        
        # 遍历所有点
        for i in range(len(shape)):
            current_point = shape[i]
            if i != index:  # 不是被修改的点
                # 如果x坐标与修改点原来的x坐标相同，且y坐标不同，且还未修改过x坐标
                if abs(current_point.x() - original_x) < 0.1 and abs(current_point.y() - original_y) >= 0.1 and not x_modified:
                    shape.moveVertexBy(i, QtCore.QPointF(dx, 0))
                    x_modified = True
                # 如果y坐标与修改点原来的y坐标相同，且x坐标不同，且还未修改过y坐标
                if abs(current_point.y() - original_y) < 0.1 and abs(current_point.x() - original_x) >= 0.1 and not y_modified:
                    shape.moveVertexBy(i, QtCore.QPointF(0, dy))
                    y_modified = True
        
        # 最后移动被修改的点
        shape.moveVertexBy(index, QtCore.QPointF(dx, dy))



    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
            logger.info(f"删除了 {len(deleted_shapes)} 个形状。")
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def paintEvent(self, event):
        # if not self.pixmap:
        if not self.pixmap or not self.drawing_enabled:
            if self.pixmap:
                p = self._painter
                p.begin(self)
                p.setRenderHint(QtGui.QPainter.Antialiasing)
                p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
                p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

                p.scale(self.scale, self.scale)
                p.translate(self.offsetToCenter())

                p.drawPixmap(0, 0, self.pixmap)

                p.scale(1 / self.scale, 1 / self.scale)
                if self.selecting and self.select_rect:
                    p = self._painter
                    p.begin(self)
                    p.setPen(QtGui.QPen(QtCore.Qt.blue, 2, QtCore.Qt.DashLine))
                    p.setBrush(QtGui.QBrush(QtCore.Qt.transparent))
                    
                    # 创建缩放后的矩形
                    scaled_rect = QtCore.QRectF(
                        self.select_rect.left() * self.scale,
                        self.select_rect.top() * self.scale,
                        self.select_rect.width() * self.scale,
                        self.select_rect.height() * self.scale
                    )
                    
                    p.drawRect(scaled_rect)
                p.end()
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        p.scale(1 / self.scale, 1 / self.scale)

        # draw crosshair
        if (
            self._crosshair[self._createMode]
            and self.drawing()
            and self.prevMovePoint
            and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 0, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y() * self.scale),
                self.width() - 1,
                int(self.prevMovePoint.y() * self.scale),
            )
            p.drawLine(
                int(self.prevMovePoint.x() * self.scale),
                0,
                int(self.prevMovePoint.x() * self.scale),
                self.height() - 1,
            )

        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            assert len(self.line.points) == len(self.line.point_labels)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (
            self.fillDrawing()
            and self.createMode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            if drawing_shape.fill_color.getRgb()[3] == 0:
                logger.warning(
                    "fill_drawing=true, but fill_color is transparent,"
                    " so forcing to be opaque."
                )
                drawing_shape.fill_color.setAlpha(64)
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)
        elif self.createMode == "ai_polygon" and self.current is not None:
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            points = self._ai_model.predict_polygon_from_points(
                points=[[point.x(), point.y()] for point in drawing_shape.points],
                point_labels=drawing_shape.point_labels,
            )
            if len(points) > 2:
                drawing_shape.setShapeRefined(
                    shape_type="polygon",
                    points=[QtCore.QPointF(point[0], point[1]) for point in points],
                    point_labels=[1] * len(points),
                )
                drawing_shape.fill = self.fillDrawing()
                drawing_shape.selected = True
                drawing_shape.paint(p)
        elif self.createMode == "ai_mask" and self.current is not None:
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            mask = self._ai_model.predict_mask_from_points(
                points=[[point.x(), point.y()] for point in drawing_shape.points],
                point_labels=drawing_shape.point_labels,
            )
            y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
            drawing_shape.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1 : y2 + 1, x1 : x2 + 1],
            )
            drawing_shape.selected = True
            drawing_shape.paint(p)

        # 绘制框选矩形
        if self.selecting and self.select_rect:
            p = self._painter
            p.begin(self)
            p.setPen(QtGui.QPen(QtCore.Qt.blue, 2, QtCore.Qt.DashLine))
            p.setBrush(QtGui.QBrush(QtCore.Qt.transparent))
            # 绘制矩形时考虑缩放
            # 创建缩放后的矩形
            scaled_rect = QtCore.QRectF(
                self.select_rect.left() * self.scale,
                self.select_rect.top() * self.scale,
                self.select_rect.width() * self.scale,
                self.select_rect.height() * self.scale
            )
            p.drawRect(scaled_rect)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        if self.createMode == "ai_polygon":
            # convert points to polygon by an AI model
            assert self.current.shape_type == "points"
            points = self._ai_model.predict_polygon_from_points(
                points=[[point.x(), point.y()] for point in self.current.points],
                point_labels=self.current.point_labels,
            )
            self.current.setShapeRefined(
                points=[QtCore.QPointF(point[0], point[1]) for point in points],
                point_labels=[1] * len(points),
                shape_type="polygon",
            )
        elif self.createMode == "ai_mask":
            # convert points to mask by an AI model
            assert self.current.shape_type == "points"
            mask = self._ai_model.predict_mask_from_points(
                points=[[point.x(), point.y()] for point in self.current.points],
                point_labels=self.current.point_labels,
            )
            y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1 : y2 + 1, x1 : x2 + 1],
            )
        self.current.close()

        self.shapes.append(self.current)
        self.storeShapes()
        logger.info(f"最终确定形状: {self.current.shape_type}，标签: {self.current.label}。")
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.setEditing(True)  # 确保切换到编辑模式
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text

        # 检查当前创建模式
        if self.createMode == "line":
            # 如果是创建直线模式，不能设置关键点标签
            if text not in self.line_labels:
                # 弹窗提示
                QMessageBox.warning(self, "错误", "只能为直线设置 line 或 entrance_line 标签，已自动更改为 line 语义标签 ！")
                # 自动更改为线标签
                text = self.line_labels[0]  # 或者根据需要选择合适的线标签
        elif self.createMode == "point":
            # 如果是创建控制点模式，只能设置关键点标签
            if text not in self.keypoint_labels:
                # 弹窗提示
                QMessageBox.warning(self, "错误", "只能为控制点设置关键点标签，已自动更改为 keypoint 语义标签 ！")
                # 自动更改为第一个关键点标签
                text = self.keypoint_labels[0]  # 或者根据需要选择合适的关键点标签
            
        # 创建标签
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        logger.info(f"设置标签为 '{text}'，标记状态为 {flags}。")
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if self._ai_model:
            self._ai_model.set_image(
                image=labelme.utils.img_qt_to_arr(self.pixmap.toImage())
            )
        if clear_shapes:
            self.shapes = []
        self.update()
    
    def setOriginalImage(self, image):
        self.original_image = image  # 设置原图
        self.update()  # 更新画布

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()

    def findParentScrollArea(self):
       parent = self.parent()
       while parent is not None:
           if isinstance(parent, QtWidgets.QScrollArea):
               return parent
           parent = parent.parent()
       return None

    def mousePressEvent(self, ev):
        # 调用 MainWindow 中的逻辑
        # self.parent().mousePressEvent(ev)
        """重写鼠标按下事件"""
        if not self._select_mode:
            # 在移动模式下，改变鼠标样式并记录起始位置
            self.setCursor(CURSOR_MOVE)
            self._pan_start_pos = ev.pos()
        else:
            # 在选择模式下，检查是否按下 Shift 键进行框选
            if ev.modifiers() & QtCore.Qt.ShiftModifier:
                self.selecting = True
                self.select_start = self.transformPos(ev.pos())
                self.select_rect = QtCore.QRectF(self.select_start, self.select_start)
            else:
                # 在选择模式下，保持原有的选择功能
                self.ori_mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        # 调用 MainWindow 中的逻辑
        if not self._select_mode and hasattr(self, '_pan_start_pos') and ev.buttons() & QtCore.Qt.LeftButton:
            # print("=======not+++++++++")
            self.setCursor(CURSOR_MOVE)
            scroll_area = self.findParentScrollArea()
            if scroll_area:
                delta = ev.pos() - self._pan_start_pos
                scroll_area.horizontalScrollBar().setValue(
                    scroll_area.horizontalScrollBar().value() - delta.x())
                scroll_area.verticalScrollBar().setValue(
                    scroll_area.verticalScrollBar().value() - delta.y())
            self._pan_start_pos = ev.pos()
        elif not self._select_mode:
            # print("======yn++======", self._select_mode, hasattr(self, '_pan_start_pos'), ev.buttons() & QtCore.Qt.LeftButton)
            self.setCursor(CURSOR_DEFAULT)
        else:
            # print("====++++yes+++", self._select_mode, hasattr(self, '_pan_start_pos'), ev.buttons() & QtCore.Qt.LeftButton)
            # self.ori_mouseMoveEvent(ev)
            # 选择模式下，修改鼠标移动事件处理
            try:
                if QT5:
                    pos = self.transformPos(ev.localPos())
                else:
                    pos = self.transformPos(ev.posF())
            except AttributeError:
                return

            self.mouseMoved.emit(pos)
            self.prevMovePoint = pos
            self.restoreCursor()

            # 处理框选
            if self.selecting and ev.buttons() & QtCore.Qt.LeftButton:
                self.select_rect = QtCore.QRectF(self.select_start, pos).normalized()
                self.update()
                return
            
            # 处理选择和高亮，但不允许拖拽
            # 只有在未按下左键时才进行高亮处理
            if not (QtCore.Qt.LeftButton & ev.buttons()):
                self.setToolTip(self.tr("Image"))
                for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
                    index = shape.nearestVertex(pos, self.epsilon)
                    index_edge = shape.nearestEdge(pos, self.epsilon)
                    if index is not None:
                        self.prevhVertex = self.hVertex = index
                        self.prevhShape = self.hShape = shape
                        self.prevhEdge = self.hEdge
                        self.hEdge = None
                        shape.highlightVertex(index, shape.MOVE_VERTEX)
                        self.overrideCursor(CURSOR_POINT)
                        self.setToolTip(self.tr("Click to select point"))
                        self.setStatusTip(self.toolTip())
                        self.update()
                        break
                    elif shape.containsPoint(pos):
                        self.prevhVertex = self.hVertex
                        self.hVertex = None
                        self.prevhShape = self.hShape = shape
                        self.prevhEdge = self.hEdge
                        self.hEdge = None
                        self.setToolTip(self.tr("Click to select shape '%s'") % shape.label)
                        self.setStatusTip(self.toolTip())
                        self.overrideCursor(CURSOR_POINT)
                        self.update()
                        break
                else:
                    self.unHighlight()
            
             # Polygon drawing.
            if self.drawing():
                if self.createMode in ["ai_polygon", "ai_mask"]:
                    self.line.shape_type = "points"
                else:
                    self.line.shape_type = self.createMode

                self.overrideCursor(CURSOR_DRAW)
                if not self.current:
                    self.repaint()  # draw crosshair
                    return

                if self.outOfPixmap(pos):
                    # Don't allow the user to draw outside the pixmap.
                    # Project the point to the pixmap's edges.
                    pos = self.intersectionPoint(self.current[-1], pos)
                elif (
                    self.snapping
                    and len(self.current) > 1
                    and self.createMode == "polygon"
                    and self.closeEnough(pos, self.current[0])
                ):
                    # Attract line to starting point and
                    # colorise to alert the user.
                    pos = self.current[0]
                    self.overrideCursor(CURSOR_POINT)
                    self.current.highlightVertex(0, Shape.NEAR_VERTEX)
                if self.createMode in ["polygon", "linestrip"]:
                    self.line.points = [self.current[-1], pos]
                    self.line.point_labels = [1, 1]
                elif self.createMode in ["ai_polygon", "ai_mask"]:
                    self.line.points = [self.current.points[-1], pos]
                    self.line.point_labels = [
                        self.current.point_labels[-1],
                        # 0 if is_shift_pressed else 1,
                        0,
                    ]
                elif self.createMode == "rectangle":
                    self.line.points = [self.current[0], pos]
                    self.line.point_labels = [1, 1]
                    self.line.close()
                elif self.createMode == "circle":
                    self.line.points = [self.current[0], pos]
                    self.line.point_labels = [1, 1]
                    self.line.shape_type = "circle"
                elif self.createMode == "line":
                    self.line.points = [self.current[0], pos]
                    self.line.point_labels = [1, 1]
                    self.line.close()
                elif self.createMode == "point":
                    self.line.points = [self.current[0]]
                    self.line.point_labels = [1]
                    self.line.close()
                assert len(self.line.points) == len(self.line.point_labels)
                self.repaint()
                self.current.highlightClear()
                return

            # Polygon copy moving.
            if QtCore.Qt.RightButton & ev.buttons():
                if self.selectedShapesCopy and self.prevPoint:
                    self.overrideCursor(CURSOR_MOVE)
                    self.boundedMoveShapes(self.selectedShapesCopy, pos)
                    self.repaint()
                elif self.selectedShapes:
                    self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                    self.repaint()
                return

            # Polygon/Vertex moving.
            if QtCore.Qt.LeftButton & ev.buttons():
                # print(self.current_filename)
                if "Slot_" in self.current_filename:
                    if self.selectedVertex():
                        self.boundedMoveVertex(pos)
                        self.repaint()
                        self.movingShape = True
                        self.editingSaveEnable.emit(True)
                        self.update()  # 更新画布
                    elif self.selectedShapes and self.prevPoint:
                        # self.overrideCursor(CURSOR_MOVE)
                        # self.boundedMoveShapes(self.selectedShapes, pos)
                        # self.repaint()
                        # self.movingShape = True
                        pass
                    return
            
            if QtCore.Qt.LeftButton & ev.buttons():
                print(self.current_filename)
                if "2D-OD" in self.current_filename:
                    if self.selectedVertex():
                        self.boundedMoveVertex_2DOD(pos)
                        self.repaint()
                        self.movingShape = True
                        self.editingSaveEnable.emit(True)
                        self.update()  # 更新画布
                    elif self.selectedShapes and self.prevPoint:
                        self.overrideCursor(CURSOR_MOVE)
                        self.boundedMoveShapes(self.selectedShapes, pos)
                        self.repaint()
                        self.movingShape = True
                        self.editingSaveEnable.emit(True)
                        pass
                    return

            self.vertexSelected.emit(self.hVertex is not None)

    def mouseReleaseEvent(self, ev):
        # self.parent().mouseReleaseEvent(ev)
        # print("=======release=======")
        """重写鼠标释放事件"""
        if not self._select_mode:
            # 在移动模式下，恢复鼠标样式
            self.setCursor(CURSOR_DEFAULT)
            # print("=======mre======", self._select_mode)
            if hasattr(self, '_pan_start_pos'):
                del self._pan_start_pos
            # self.parent().mouseReleaseEvent(ev)
        elif self.selecting:
            if ev.button() == QtCore.Qt.LeftButton:
                # 框选结束，选择完全在框内的形状
                selected_shapes = []
                for shape in self.shapes:
                    if self.isVisible(shape):
                        # 检查形状的所有点是否都在框内
                        all_points_in_rect = True
                        for point in shape.points:
                            if not self.select_rect.contains(point):
                                all_points_in_rect = False
                                break
                        if all_points_in_rect:
                            selected_shapes.append(shape)
                
                if selected_shapes:
                    self.selectionChanged.emit(selected_shapes)
                
                self.selecting = False
                self.select_start = None
                self.select_rect = None
                self.update()
            # self.parent().mouseReleaseEvent(ev)
        else:
            # 在选择模式下，保持原有的选择功能
            # self.ori_mouseReleaseEvent(ev)
            if ev.button() == QtCore.Qt.RightButton:
                menu = self.menus[len(self.selectedShapesCopy) > 0]
                self.restoreCursor()
                if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                    # Cancel the move by deleting the shadow copy.
                    self.selectedShapesCopy = []
                    self.repaint()
            elif ev.button() == QtCore.Qt.LeftButton:
                # 选择模式下左键释放，只处理选择变更
                if self.hShape is not None and self.hShapeIsSelected:
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )
            # self.parent().mouseReleaseEvent(ev)
