class IoU(tf.keras.metrics.Metric):
      def __init__(self, **kwargs):
          super(IoU, self).__init__(**kwargs)

          self.iou = self.add_weight(name='iou', initializer='zeros')
          self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
          self.num_ex = self.add_weight(name='num_ex', initializer='zeros')

      def update_state(self, y_true, y_pred, sample_weight=None):
          def get_box(y):
              rows, cols = y[:, 0], y[:, 1]
              rows, cols = rows*144, cols*144
              y1, y2 = rows, rows + 52
              x1, x2 = cols, cols + 52
              return x1, y1, x2, y2

          def get_area(x1, y1, x2, y2):
              return tf.math.abs(x2 - x1) * tf.math.abs(y2 - y1)

          gt_x1, gt_y1, gt_x2, gt_y2 = get_box(y_true)
          p_x1, p_y1, p_x2, p_y2 = get_box(y_pred)

          i_x1 = tf.maximum(gt_x1, p_x1)
          i_y1 = tf.maximum(gt_y1, p_y1)

          i_x2 = tf.minimum(gt_x2, p_x2)
          i_y2 = tf.minimum(gt_y2, p_y2)

          i_area = get_area(i_x1, i_y1, i_x2, i_y2)
          u_area = get_area(gt_x1, gt_y1, gt_x2, gt_y2) + get_area(p_x1, p_y1, p_x2, p_y2) - i_area
          
          iou = tf.math.divide(i_area, u_area)
          self.num_ex.assign_add(1)
          self.total_iou.assign_add(tf.reduce_mean(iou))
          self.iou = tf.math.divide(self.total_iou, self.num_ex)

      def result(self):
          return self.iou

      def reset_state(self):
          self.iou = self.add_weight(name='iou', initializer='zeros')
          self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
          self.num_ex = self.add_weight(name='num_ex', initializer='zeros')