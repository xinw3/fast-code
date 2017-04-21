package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.IntWritable;

public class SimilarityReducer extends Reducer<Text, Text, Text, Text> {

	@Override
	protected void reduce(Text key, Iterable<Text> value, Context context)
			throws IOException, InterruptedException {

		int count = 0;
		for (Text val : value) {
			String str = val.toString();
			count += Integer.parseInt(str);
		}
		context.write(key, new Text(Integer.toString(count)));
	}
}
